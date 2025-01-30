import json
import os
import re
import time

import boto3
import boto3.s3.transfer as s3transfer
import botocore.client
from huggingface_hub import scan_cache_dir, snapshot_download
from tqdm import tqdm

IMPORT_MODEL_ROLE_NAME = "MyImportModelRole"


def fast_upload(
    s3_bucket: str,
    s3_prefix: str,
    local_path: str,
    files: list[str],
    progress_func: callable,
    workers: int = 20,
):
    """
    Uploads multiple files to S3 in parallel using a thread pool.

    Args:
        s3_bucket (str): Name of the S3 bucket to upload to
        s3_prefix (str): Prefix/path within the S3 bucket
        local_path (str): Local directory path containing files to upload
        files (list[str]): List of filenames to upload
        progress_func (callable): Callback function to track upload progress
        workers (int, optional): Number of concurrent upload workers. Defaults to 20.
    """
    s3client = boto3.client("s3", config=botocore.client.Config(max_pool_connections=workers))
    transfer_config = s3transfer.TransferConfig(
        use_threads=True,
        max_concurrency=workers,
    )
    s3t = s3transfer.create_transfer_manager(s3client, transfer_config)
    for src in files:
        dst = os.path.join(s3_prefix, src)
        s3t.upload(
            os.path.join(local_path, src),
            s3_bucket,
            dst,
            subscribers=[s3transfer.ProgressCallbackInvoker(progress_func)],
        )
    s3t.shutdown()


def download_model(model_id: str) -> str:
    """
    Downloads a model from Hugging Face Hub to local storage.

    Args:
        model_id (str): The Hugging Face model ID to download (e.g. "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

    Returns:
        str: Local path where the model was downloaded to

    Prints progress messages during download.
    """

    print(f"Downloading model {model_id} from Hugging Face...")
    local_path = snapshot_download(repo_id=model_id)
    print(f"Model downloaded to {local_path}.")
    return local_path


def upload_to_s3(local_path: str, s3_bucket: str, model_name: str) -> str:
    """
    Uploads model files from local path to S3 bucket.

    Args:
        local_path (str): Local directory containing model files
        s3_bucket (str): S3 bucket to upload to
        model_name (str): Model name used to create the S3 prefix

    Returns:
        str: Full S3 URI where files were uploaded

    Only uploads files that don't already exist in S3. Shows progress bar during upload.
    """

    s3_client = boto3.client("s3")

    model_prefix = f"models/{model_name}"

    upload_s3_bucket, upload_s3_key = s3_uri_to_bucket_and_key(
        f"{s3_bucket}/{model_prefix}"
    )
    upload_location = f"s3://{upload_s3_bucket}/{upload_s3_key}"
    print(f"Uploading model files to {upload_location}...")

    files_to_upload = []

    for root, dirs, files in os.walk(local_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_file_path = os.path.relpath(file_path, local_path)

            # check if file already exists on S3
            try:
                s3_client.head_object(
                    Bucket=upload_s3_bucket, Key=f"{upload_s3_key}/{relative_file_path}"
                )
            except s3_client.exceptions.ClientError as e:
                if int(e.response["Error"]["Code"]) == 404:
                    # upload only if not exists
                    files_to_upload.append(relative_file_path)

    if files_to_upload:
        totalsize = sum([os.stat(f"{local_path}/{f}").st_size for f in files_to_upload])

        with tqdm(
            desc="Upload to S3 bucket",
            dynamic_ncols=True,
            total=totalsize,
            unit="B",
            unit_scale=1,
        ) as pbar:
            fast_upload(
                upload_s3_bucket,
                upload_s3_key,
                local_path,
                files_to_upload,
                pbar.update,
            )
    else:
        print("All model files already uploaded.")

    return upload_location


def get_bedrock_import_role(s3_bucket: str) -> str:
    """
    Creates an IAM role for Bedrock model import if it doesn't exist.

    Args:
        s3_bucket (str): S3 bucket where model files are uploaded, to give permission to the IAM role

    Returns:
        str: ARN of the IAM role
    """
    iam = boto3.client("iam")

    try:
        role_response = iam.get_role(RoleName=IMPORT_MODEL_ROLE_NAME)
        return role_response["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        #print("Role does not exist, creating...")
        pass

    create_role_response = iam.create_role(
        RoleName=IMPORT_MODEL_ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "bedrock.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }
        ),
    )

    iam.put_role_policy(
        RoleName=create_role_response["Role"]["RoleName"],
        PolicyName=f"{create_role_response['Role']['RoleName']}S3BucketPolicy",
        PolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:ListBucket"],
                        "Resource": [
                            f"arn:aws:s3:::{s3_bucket}",
                            f"arn:aws:s3:::{s3_bucket}/*",
                        ],
                    }
                ],
            }
        ),
    )

    return create_role_response["Role"]["Arn"]


def s3_uri_to_bucket_and_key(s3_uri: str) -> tuple[str, str]:
    """
    Extracts the bucket name and S3 key from an S3 URI.

    Args:
        s3_uri (str): S3 URI

    Returns:
        tuple[str, str]: S3 bucket name and S3 key
    """
    return os.path.normpath(s3_uri.removeprefix("s3://")).split("/", maxsplit=1)


def sanitize_model_id(model_id: str) -> str:
    """
    Sanitizes the model ID to be used as a folder name.

    Args:
        model_id (str): Model ID

    Returns:
        str: Sanitized model ID
    """
    return re.sub(r'[^-a-zA-Z0-9]', "-", model_id)


def create_bedrock_model(s3_uri: str, model_name: str, role_arn: str) -> str:
    """
    Creates a model import job in Amazon Bedrock.

    Args:
        s3_uri (str): S3 URI containing model files
        model_name (str): Model name to use for naming
        role_arn (str): ARN of IAM role to use for model import

    Returns:
        str: ARN of the imported model

    Waits for import job completion and handles errors.
    """

    bedrock = boto3.client("bedrock")

    # first, check if the model already exists
    try:
        model_response = bedrock.get_imported_model(modelIdentifier=model_name)
        print(f"Model {model_name} already exists, skipping import.")
        model_arn = model_response["modelArn"]
        print(f"Imported model ARN: {model_arn}")
        return model_arn
    except (bedrock.exceptions.ResourceNotFoundException, bedrock.exceptions.ClientError):
        # model does not exist, continue
        pass

    job_name = f"import-{model_name[:20]}-{time.time_ns()}"
    print(f"Creating Bedrock model import job with name {job_name}...")

    create_job_response = bedrock.create_model_import_job(
        jobName=job_name,
        importedModelName=model_name,
        roleArn=role_arn,
        modelDataSource={"s3DataSource": {"s3Uri": s3_uri}},
    )

    job_arn = create_job_response.get("jobArn")

    wait_secs = 0
    sleep_time = 10
    print(
        "Waiting while model import job is in progress (depending on the model size this may take some time)..",
        end="",
    )
    while wait_secs < 3600:  # wait max 1 hour
        print(".", end="")
        time.sleep(sleep_time)
        wait_secs += sleep_time

        job_response = bedrock.get_model_import_job(jobIdentifier=job_arn)
        if job_response["status"] != "InProgress":
            print("")
            if "failureMessage" in job_response:
                print(f"Failed to import model: {job_response['failureMessage']}")
                raise Exception(
                    f"Failed to import model: {job_response['failureMessage']}"
                )
            break

    print(f"Finished model import with status '{job_response['status']}'")
    model_arn = job_response["importedModelArn"]
    print(f"Imported model ARN: {model_arn}")
    return model_arn


def test_model(model_arn: str):
    """
    Tests an imported Bedrock model with a sample prompt.

    Args:
        model_arn (str): ARN of the Bedrock model to test

    Prints the model response or any errors that occur.
    """
    print("Testing the model...")
    bedrock_runtime = boto3.client(
        "bedrock-runtime", config=botocore.client.Config(retries={"max_attempts": 10})
    )

    prompt = "What is the capital of France?"

    invoke_response = bedrock_runtime.invoke_model(
        modelId=model_arn, body=json.dumps({"prompt": prompt})
    )
    invoke_response["body"] = json.loads(invoke_response["body"].read().decode("utf-8"))
    print(json.dumps(invoke_response, indent=4))


def cleanup_aws_resources(s3_uri: str, model_id: str):
    """
    Cleans up resources created during model import.

    Args:
        s3_bucket (str): S3 bucket to clean
        model_id (str): Model ID to delete

    Deletes the Bedrock model, S3 contents, and IAM role.
    """
    print("Cleaning up resources...")

    # Delete the iam role
    iam = boto3.client("iam")
    try:
        iam.delete_role_policy(
            RoleName=IMPORT_MODEL_ROLE_NAME,
            PolicyName="MyImportModelRoleS3BucketPolicy",
        )
        iam.delete_role(RoleName=IMPORT_MODEL_ROLE_NAME)
        print("Deleted role.")
    except Exception as e:
        print(f"Unable to delete role: {repr(e)}")

    # Delete the model
    bedrock = boto3.client("bedrock")
    try:
        bedrock.delete_custom_model(modelIdentifier=model_id)
        print("Deleted Bedrock custom model.")
    except Exception as e:
        print(f"Unable to delete Bedrock custom model: {repr(e)}")

    # Delete S3 contents
    s3 = boto3.resource("s3")
    try:
        bucket, prefix = s3_uri_to_bucket_and_key(s3_uri)
        prefix = os.path.normpath(f"{prefix}/models/{model_id}")
        s3.Bucket(bucket).objects.filter(Prefix=prefix).delete()
        print("Deleted S3 prefix with model files.")
    except Exception as e:
        print(f"Unable to delete S3 prefix: {repr(e)}")

    # Remove local files
    # if os.path.exists(local_path):
    #    shutil.rmtree(local_path)

def cleanup_local_resources(model_id: str):
    """
    Cleans up resources created during model import.

    Args:
        model_id (str): Model ID to delete

    Deletes the local files.
    """
    print("Cleaning up local model resources...")

    scan_result = scan_cache_dir()
    delete_strategy = None
    for repo in scan_result.repos:
        if repo.repo_id == model_id:
            print("Model found in cache, deleting it.")
            delete_strategy = scan_result.delete_revisions([x.commit_hash for x in repo.revisions])
            break

    if not delete_strategy:
        print("Model not found in cache.")
    else:
        print("Model cleanup will free " + delete_strategy.expected_freed_size_str)
        delete_strategy.execute()


def import_model_to_bedrock(model_id: str, s3_bucket: str) -> str:
    """
    Import a Hugging Face model to Amazon Bedrock.
    
    Args:
        model_id (str): Hugging Face model ID
        s3_bucket (str): S3 bucket to upload the model to
        
    Returns:
        model_arn: The ARN of the created Bedrock model
    """
    # Step 1: Download model from Hugging Face
    local_path = download_model(model_id)

    model_name = sanitize_model_id(model_id)

    # Step 2: Upload to S3
    s3_uri = upload_to_s3(local_path, s3_bucket, model_name)

    # Step 3: Create Bedrock model (with IAM role for import)
    role_arn = get_bedrock_import_role(
        s3_bucket=s3_uri_to_bucket_and_key(s3_uri)[0]
    )
    model_arn = create_bedrock_model(s3_uri, model_name, role_arn)

    ## Step 4: Test the model
    #test_model(model_arn)
    
    return model_arn
