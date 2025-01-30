import argparse
import sys
import urllib

import boto3

from huggingface_bedrock_importer.importer import (
    cleanup_aws_resources,
    cleanup_local_resources,
    import_model_to_bedrock,
    test_model,
)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def validate_aws_credentials():
    try:
        client = boto3.client("sts")
        client.get_caller_identity()
        print(
            f"{bcolors.HEADER}Using AWS region {client.meta.region_name}. Make sure all functionality is available there.{bcolors.ENDC}"
        )
    except Exception as e:
        print(
            f"{bcolors.FAIL}Unable to locate AWS credentials, please configure them: {e}{bcolors.ENDC}"
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download and deploy a Hugging Face model to AWS Bedrock"
    )
    parser.add_argument(
        "--model-id",
        help="Hugging Face model ID",
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        required=False,
    )
    parser.add_argument(
        "--s3-uri",
        help="S3 URI for model storage. Can be a bucket name only, or a prefix (e.g., s3://amzn-s3-demo-bucket/my_models)",
        required=False,
    )
    parser.add_argument(
        "--cleanup-resources",
        action="store_true",
        help="Cleanup AWS resources (Bedrock custom model, IAM role, S3 model files)",
    )
    parser.add_argument(
        "--cleanup-model",
        action="store_true",
        help="Cleanup local model files",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the model after importing it",
    )

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    ran_cleanup = False
    if args.cleanup_resources:
        print(
            f"{bcolors.WARNING}Destroying remote entities (Bedrock custom model, IAM role, S3 model files)...{bcolors.ENDC}"
        )
        if not args.model_id or not args.s3_uri:
            print(
                f"{bcolors.FAIL}Please provide both model_id and s3_uri to cleanup resources.{bcolors.ENDC}"
            )
            sys.exit(1)
        validate_aws_credentials()
        cleanup_aws_resources(args.s3_uri, args.model_id)
        print(f"{bcolors.OKGREEN}Done.{bcolors.ENDC}")
        ran_cleanup = True

    if args.cleanup_model:
        print(f"{bcolors.WARNING}Cleaning up local model resources...{bcolors.ENDC}")
        if not args.model_id:
            print(
                f"{bcolors.FAIL}Please provide model_id to cleanup resources.{bcolors.ENDC}"
            )
            sys.exit(1)
        cleanup_local_resources(args.model_id)
        print(f"{bcolors.OKGREEN}Done.{bcolors.ENDC}")
        ran_cleanup = True

    if not ran_cleanup:
        if not args.model_id or not args.s3_uri:
            print(
                f"{bcolors.FAIL}Please provide both model_id and s3_uri to import a model.{bcolors.ENDC}"
            )
            sys.exit(1)
        print(f"{bcolors.HEADER}Using model ID: {args.model_id}{bcolors.ENDC}")
        print(f"{bcolors.HEADER}Using S3 location: {args.s3_uri}{bcolors.ENDC}")

        try:
            validate_aws_credentials()
            model_arn = import_model_to_bedrock(args.model_id, args.s3_uri)

            print(f"{bcolors.BOLD}Process completed successfully!{bcolors.ENDC}")
            print(
                f"{bcolors.OKGREEN}Link to the Bedrock playground for the model: https://{model_arn.split(':')[3]}.console.aws.amazon.com/bedrock/home#/text-generation-playground?mode=text&modelId={urllib.parse.quote_plus(model_arn)}{bcolors.ENDC}"
            )

            # Step 4: Test the model
            if args.test:
                print("Testing the model...")
                test_model(model_arn)

        except Exception as e:
            print(f"{bcolors.FAIL}An error occurred: {repr(e)}{bcolors.ENDC}")
            raise e


if __name__ == "__main__":
    main()
