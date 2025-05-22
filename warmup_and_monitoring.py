import boto3
import json
import time
import os
from datetime import datetime, timedelta
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

lambda_client = boto3.client('lambda')
cloudwatch = boto3.client('cloudwatch')

def warmup_handler(event, context):
    if event.get('warmup'):
        logger.info("Lambda warmup request received")
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Lambda warmed up successfully',
                'timestamp': int(time.time())
            })
        }
    
    function_name = os.environ.get('CHATBOT_FUNCTION_NAME')
    
    if not function_name:
        logger.error("CHATBOT_FUNCTION_NAME environment variable not set")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Function name not configured'})
        }
    
    try:
        warmup_payload = {
            'httpMethod': 'POST',
            'body': json.dumps({
                'query': 'warmup test query'
            }),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
        
        start_time = time.time()
        
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(warmup_payload)
        )
        
        end_time = time.time()
        warmup_latency = (end_time - start_time) * 1000
        
        response_payload = json.loads(response['Payload'].read())
        
        cloudwatch.put_metric_data(
            Namespace='AcademicChatbot/Warmup',
            MetricData=[
                {
                    'MetricName': 'WarmupLatency',
                    'Value': warmup_latency,
                    'Unit': 'Milliseconds',
                    'Timestamp': datetime.utcnow()
                },
                {
                    'MetricName': 'WarmupSuccess',
                    'Value': 1 if response['StatusCode'] == 200 else 0,
                    'Unit': 'Count',
                    'Timestamp': datetime.utcnow()
                }
            ]
        )
        
        logger.info(f"Warmup completed in {warmup_latency:.2f}ms")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Warmup completed successfully',
                'latency_ms': warmup_latency,
                'function_status': response['StatusCode'],
                'timestamp': int(time.time())
            })
        }
        
    except Exception as e:
        logger.error(f"Warmup failed: {str(e)}")
        
        cloudwatch.put_metric_data(
            Namespace='AcademicChatbot/Warmup',
            MetricData=[
                {
                    'MetricName': 'WarmupSuccess',
                    'Value': 0,
                    'Unit': 'Count',
                    'Timestamp': datetime.utcnow()
                }
            ]
        )
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Warmup failed',
                'message': str(e),
                'timestamp': int(time.time())
            })
        }

def performance_monitor_handler(event, context):
    try:
        function_name = os.environ.get('CHATBOT_FUNCTION_NAME')
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)
        
        metrics_response = cloudwatch.get_metric_statistics(
            Namespace='AWS/Lambda',
            MetricName='Duration',
            Dimensions=[
                {
                    'Name': 'FunctionName',
                    'Value': function_name
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=['Average', 'Maximum', 'Minimum']
        )
        
        invocation_response = cloudwatch.get_metric_statistics(
            Namespace='AWS/Lambda',
            MetricName='Invocations',
            Dimensions=[
                {
                    'Name': 'FunctionName',
                    'Value': function_name
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=['Sum']
        )
        
        error_response = cloudwatch.get_metric_statistics(
            Namespace='AWS/Lambda',
            MetricName='Errors',
            Dimensions=[
                {
                    'Name': 'FunctionName',
                    'Value': function_name
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=['Sum']
        )
        
        performance_data = {
            'function_name': function_name,
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'duration_metrics': {
                'datapoints': len(metrics_response['Datapoints']),
                'average_ms': round(metrics_response['Datapoints'][0]['Average'], 2) if metrics_response['Datapoints'] else 0,
                'max_ms': round(metrics_response['Datapoints'][0]['Maximum'], 2) if metrics_response['Datapoints'] else 0,
                'min_ms': round(metrics_response['Datapoints'][0]['Minimum'], 2) if metrics_response['Datapoints'] else 0
            },
            'invocation_count': sum(dp['Sum'] for dp in invocation_response['Datapoints']),
            'error_count': sum(dp['Sum'] for dp in error_response['Datapoints']),
            'timestamp': int(time.time())
        }
        
        if performance_data['invocation_count'] > 0:
            performance_data['error_rate'] = round(
                (performance_data['error_count'] / performance_data['invocation_count']) * 100, 2
            )
        else:
            performance_data['error_rate'] = 0
        
        cloudwatch.put_metric_data(
            Namespace='AcademicChatbot/Performance',
            MetricData=[
                {
                    'MetricName': 'AverageLatency',
                    'Value': performance_data['duration_metrics']['average_ms'],
                    'Unit': 'Milliseconds',
                    'Timestamp': datetime.utcnow()
                },
                {
                    'MetricName': 'ErrorRate',
                    'Value': performance_data['error_rate'],
                    'Unit': 'Percent',
                    'Timestamp': datetime.utcnow()
                }
            ]
        )
        
        logger.info(f"Performance monitoring completed for {function_name}")
        
        return {
            'statusCode': 200,
            'body': json.dumps(performance_data)
        }
        
    except Exception as e:
        logger.error(f"Performance monitoring failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Performance monitoring failed',
                'message': str(e),
                'timestamp': int(time.time())
            })
        }