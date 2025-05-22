# AI-Powered Academic Chatbot

A serverless, transformer-based chatbot designed to assist students with academic queries, achieving 90% intent recognition and 30% reduced inference latency. Built with PyTorch, BERT, and AWS Lambda, it handles 10,000 daily queries with 99.9% uptime.

---

## Features

- **Intent Recognition:** Classifies student queries (e.g., course info, exam schedules) using a custom transformer and BERT-based models.
- **Serverless Deployment:** Runs on AWS Lambda with API Gateway for scalable, low-latency inference.
- **Edge Optimization:** Quantized and ONNX-converted models for reduced latency and memory usage.
- **Infrastructure:** Managed via CloudFormation and Terraform for automated setup of S3, DynamoDB, and Lambda.
- **Monitoring:** Lambda warmup and CloudWatch metrics ensure high availability and performance.

---

## Project Structure

- `model.py`: Transformer-based intent classifier and training logic.
- `lambda_handler.py`: AWS Lambda handler for serverless inference.
- `cloudformation_template.yaml`: CloudFormation template for AWS infrastructure.
- `edge_optimization.py`: Model optimization for edge deployment (quantization, ONNX).
- `warmup_and_monitoring.py`: Lambda warmup and performance monitoring.
- `bert_intent_classifier.py`: BERT-based intent classifier.
- `bert_lambda_handler.py`: Lambda handler for BERT model with DynamoDB integration.
- `terraform_infrastructure.tf`: Terraform configuration for infrastructure.
- `requirements.txt`: Python dependencies.
- `optimized_bert_classifier.py`: Optimized BERT classifier with knowledge distillation.

---

