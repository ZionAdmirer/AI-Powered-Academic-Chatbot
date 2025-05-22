terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "academic-chatbot"
}

resource "aws_s3_bucket" "model_bucket" {
  bucket = "${var.project_name}-models-${random_id.bucket_suffix.hex}"
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_versioning" "model_bucket_versioning" {
  bucket = aws_s3_bucket.model_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_dynamodb_table" "responses_table" {
  name           = "${var.project_name}-responses"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "intent"

  attribute {
    name = "intent"
    type = "S"
  }

  tags = {
    Name = "${var.project_name}-responses"
  }
}

resource "aws_dynamodb_table" "interactions_table" {
  name           = "${var.project_name}-interactions"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "interaction_id"

  attribute {
    name = "interaction_id"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "N"
  }

  global_secondary_index {
    name     = "timestamp-index"
    hash_key = "timestamp"
  }

  tags = {
    Name = "${var.project_name}-interactions"
  }
}

resource "aws_iam_role" "lambda_execution_role" {
  name = "${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "${var.project_name}-lambda-policy"
  role = aws_iam_role.lambda_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "${aws_s3_bucket.model_bucket.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:Query",
          "dynamodb:Scan"
        ]
        Resource = [
          aws_dynamodb_table.responses_table.arn,
          aws_dynamodb_table.interactions_table.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "lambda:InvokeFunction"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_lambda_function" "chatbot_function" {
  filename         = "chatbot_deployment.zip"
  function_name    = "${var.project_name}-main"
  role            = aws_iam_role.lambda_execution_role.arn
  handler         = "lambda_handler.lambda_handler"
  runtime         = "python3.9"
  timeout         = 30
  memory_size     = 1024

  environment {
    variables = {
      MODEL_BUCKET     = aws_s3_bucket.model_bucket.bucket
      RESPONSES_TABLE  = aws_dynamodb_table.responses_table.name
      INTERACTIONS_TABLE = aws_dynamodb_table.interactions_table.name
    }
  }

  depends_on = [aws_iam_role_policy.lambda_policy]
}

resource "aws_lambda_function" "warmup_function" {
  filename         = "warmup_deployment.zip"
  function_name    = "${var.project_name}-warmup"
  role            = aws_iam_role.lambda_execution_role.arn
  handler         = "warmup_handler.warmup_handler"
  runtime         = "python3.9"
  timeout         = 60
  memory_size     = 256

  environment {
    variables = {
      CHATBOT_FUNCTION_NAME = aws_lambda_function.chatbot_function.function_name
    }
  }

  depends_on = [aws_iam_role_policy.lambda_policy]
}

resource "aws_lambda_function" "performance_monitor" {
  filename         = "warmup_deployment.zip"
  function_name    = "${var.project_name}-performance-monitor"
  role            = aws_iam_role.lambda_execution_role.arn
  handler         = "warmup_handler.performance_monitor_handler"
  runtime         = "python3.9"
  timeout         = 60
  memory_size     = 256

  environment {
    variables = {
      CHATBOT_FUNCTION_NAME = aws_lambda_function.chatbot_function.function_name
    }
  }

  depends_on = [aws_iam_role_policy.lambda_policy]
}

resource "aws_api_gateway_rest_api" "chatbot_api" {
  name        = "${var.project_name}-api"
  description = "Academic Chatbot API"

  endpoint_configuration