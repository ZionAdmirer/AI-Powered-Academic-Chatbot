import json
import boto3
import os
import time
import logging
from datetime import datetime
import torch
from intent_classifier import IntentPredictor, preprocess_text

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
cloudwatch = boto3.client('cloudwatch')

INTENT_PREDICTOR = None
RESPONSES_TABLE = None

def cold_start_init():
    global INTENT_PREDICTOR, RESPONSES_TABLE
    
    if INTENT_PREDICTOR is None:
        model_bucket = os.environ.get('MODEL_BUCKET')
        model_key = os.environ.get('MODEL_KEY', 'models/intent_classifier.pth')
        tokenizer_key = os.environ.get('TOKENIZER_KEY', 'models/tokenizer')
        labels_key = os.environ.get('LABELS_KEY', 'models/intent_labels.json')
        
        local_model_path = '/tmp/intent_classifier.pth'
        local_tokenizer_path = '/tmp/tokenizer'
        local_labels_path = '/tmp/intent_labels.json'
        
        s3_client.download_file(model_bucket, model_key, local_model_path)
        s3_client.download_file(model_bucket, tokenizer_key, local_tokenizer_path)
        s3_client.download_file(model_bucket, labels_key, local_labels_path)
        
        INTENT_PREDICTOR = IntentPredictor(
            model_path=local_model_path,
            tokenizer_path=local_tokenizer_path,
            intent_labels_path=local_labels_path
        )
    
    if RESPONSES_TABLE is None:
        table_name = os.environ.get('RESPONSES_TABLE', 'academic-chatbot-responses')
        RESPONSES_TABLE = dynamodb.Table(table_name)

def get_response_template(intent):
    try:
        response = RESPONSES_TABLE.get_item(Key={'intent': intent})
        if 'Item' in response:
            return response['Item']['template']
        else:
            return "I understand you're asking about {}, but I don't have specific information on that topic."
    except Exception as e:
        logger.error(f"Error fetching response template: {str(e)}")
        return "I'm sorry, but I'm having trouble processing your request right now."

def generate_response(intent, query, confidence):
    template = get_response_template(intent)
    
    response_data = {
        'response': template.format(query),
        'intent': intent,
        'confidence': confidence,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    return response_data

def log_interaction(query, intent, confidence, response_time):
    try:
        cloudwatch.put_metric_data(
            Namespace='AcademicChatbot/Interactions',
            MetricData=[
                {
                    'MetricName': 'ResponseTime',
                    'Value': response_time,
                    'Unit': 'Milliseconds',
                    'Dimensions': [
                        {
                            'Name': 'Intent',
                            'Value': intent
                        }
                    ]
                },
                {
                    'MetricName': 'IntentConfidence',
                    'Value': confidence,
                    'Unit': 'None',
                    'Dimensions': [
                        {
                            'Name': 'Intent',
                            'Value': intent
                        }
                    ]
                },
                {
                    'MetricName': 'QueryCount',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [
                        {
                            'Name': 'Intent',
                            'Value': intent
                        }
                    ]
                }
            ]
        )
    except Exception as e:
        logger.error(f"Error logging metrics: {str(e)}")

def lambda_handler(event, context):
    start_time = time.time()
    
    try:
        cold_start_init()
        
        if event.get('httpMethod') == 'POST':
            body = json.loads(event.get('body', '{}'))
            query = body.get('query', '')
        else:
            query = event.get('query', '')
        
        if not query:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'Query parameter is required'
                })
            }
        
        processed_query = preprocess_text(query)
        prediction = INTENT_PREDICTOR.predict(processed_query)
        
        intent = prediction['intent']
        confidence = prediction['confidence']
        
        response_data = generate_response(intent, query, confidence)
        
        end_time = time.time()
        response_time = (end_time - start_time) * 1000
        
        log_interaction(query, intent, confidence, response_time)
        
        response_data['response_time_ms'] = response_time
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response_data)
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        
        end_time = time.time()
        response_time = (end_time - start_time) * 1000
        
        cloudwatch.put_metric_data(
            Namespace='AcademicChatbot/Errors',
            MetricData=[
                {
                    'MetricName': 'ErrorCount',
                    'Value': 1,
                    'Unit': 'Count'
                },
                {
                    'MetricName': 'ErrorResponseTime',
                    'Value': response_time,
                    'Unit': 'Milliseconds'
                }
            ]
        )
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'response_time_ms': response_time
            })
        }