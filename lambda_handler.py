import json
import torch
import boto3
import os
from io import BytesIO
import pickle
import time
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')
MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'academic-chatbot-models')
MODEL_KEY = 'academic_chatbot_model.pth'

model_cache = {}

def download_model_from_s3():
    try:
        response = s3_client.get_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
        model_data = response['Body'].read()
        return torch.load(BytesIO(model_data), map_location='cpu')
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

def load_model():
    global model_cache
    
    if 'model' not in model_cache:
        logger.info("Loading model for the first time...")
        start_time = time.time()
        
        checkpoint = download_model_from_s3()
        
        from model import IntentClassifier, TextProcessor
        
        processor = checkpoint['processor']
        label_encoder = checkpoint['label_encoder']
        config = checkpoint['model_config']
        
        model = IntentClassifier(
            vocab_size=config['vocab_size'],
            n_classes=config['n_classes'],
            d_model=128,
            n_heads=4,
            n_layers=3
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        model_cache['model'] = model
        model_cache['processor'] = processor
        model_cache['label_encoder'] = label_encoder
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    return model_cache['model'], model_cache['processor'], model_cache['label_encoder']

def predict_intent(text, model, processor, label_encoder):
    start_time = time.time()
    
    with torch.no_grad():
        encoded_text = processor.encode(text).unsqueeze(0)
        output = model(encoded_text)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()
        
        intent = label_encoder.inverse_transform([predicted_class])[0]
    
    inference_time = time.time() - start_time
    logger.info(f"Inference completed in {inference_time:.4f} seconds")
    
    return intent, confidence, inference_time

def generate_response(intent, confidence):
    responses = {
        "course_info": {
            "message": "I can help you find course information. Please specify which course you're interested in, and I'll provide details about prerequisites, schedule, and requirements.",
            "quick_actions": ["View Course Catalog", "Check Prerequisites", "See Schedule"]
        },
        "assignment_help": {
            "message": "I'm here to assist with your assignments. What specific help do you need? I can guide you through requirements, deadlines, and submission processes.",
            "quick_actions": ["View Assignment", "Check Due Date", "Submit Work"]
        },
        "exam_schedule": {
            "message": "Let me help you with exam scheduling information. Which course exam are you asking about? I can provide dates, times, and locations.",
            "quick_actions": ["View Exam Schedule", "Set Reminder", "Find Study Group"]
        },
        "grades_inquiry": {
            "message": "I can help you with grade-related questions. Please check your student portal for detailed grades, or I can guide you through the process.",
            "quick_actions": ["View Grades", "Request Transcript", "Grade Appeal"]
        },
        "registration": {
            "message": "For registration assistance, I can help you navigate course selection, prerequisites, and scheduling. What would you like to register for?",
            "quick_actions": ["Add Course", "Drop Course", "View Schedule"]
        },
        "financial_aid": {
            "message": "I can assist with financial aid information including scholarships, grants, and loan options. What specific information do you need?",
            "quick_actions": ["Check Aid Status", "Apply for Aid", "Payment Options"]
        },
        "library_hours": {
            "message": "The library hours vary by location and season. Main library is typically open 24/7 during the semester. Would you like specific branch information?",
            "quick_actions": ["View All Hours", "Book Study Room", "Library Services"]
        },
        "technical_support": {
            "message": "I can help with technical issues including Wi-Fi, learning management system, and student portal problems. What technical issue are you experiencing?",
            "quick_actions": ["WiFi Help", "Portal Issues", "LMS Support"]
        }
    }
    
    if confidence > 0.75:
        response_data = responses.get(intent, {
            "message": "I understand your question but need more specific information to provide the best help.",
            "quick_actions": ["Contact Support", "Browse FAQ", "Live Chat"]
        })
    elif confidence > 0.5:
        response_data = {
            "message": f"I think you're asking about {intent.replace('_', ' ')}. Could you provide more details so I can give you accurate information?",
            "quick_actions": ["Clarify Question", "Browse Topics", "Contact Advisor"]
        }
    else:
        response_data = {
            "message": "I'm not sure I understand your question completely. Could you please rephrase it? You can also browse our help topics or contact student services directly.",
            "quick_actions": ["Rephrase Question", "Browse Help", "Contact Support"]
        }
    
    return response_data

def lambda_handler(event, context):
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        if event.get('httpMethod') == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS'
                },
                'body': ''
            }
        
        body = json.loads(event.get('body', '{}'))
        user_query = body.get('query', '').strip()
        
        if not user_query:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({
                    'error': 'Query parameter is required',
                    'message': 'Please provide a query in the request body'
                })
            }
        
        model, processor, label_encoder = load_model()
        
        intent, confidence, inference_time = predict_intent(
            user_query, model, processor, label_encoder
        )
        
        response_data = generate_response(intent, confidence)
        
        result = {
            'query': user_query,
            'intent': intent,
            'confidence': round(confidence, 3),
            'response': response_data['message'],
            'quick_actions': response_data['quick_actions'],
            'processing_time_ms': round(inference_time * 1000, 2),
            'timestamp': int(time.time())
        }
        
        logger.info(f"Processed query successfully: {intent} (confidence: {confidence:.3f})")
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps(result)
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': 'An error occurred while processing your request. Please try again.',
                'timestamp': int(time.time())
            })
        }

def health_check_handler(event, context):
    try:
        model, processor, label_encoder = load_model()
        
        test_query = "Hello, can you help me?"
        intent, confidence, inference_time = predict_intent(
            test_query, model, processor, label_encoder
        )
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'status': 'healthy',
                'model_loaded': True,
                'test_inference_time_ms': round(inference_time * 1000, 2),
                'timestamp': int(time.time())
            })
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': int(time.time())
            })
        }

def batch_process_handler(event, context):
    try:
        queries = event.get('queries', [])
        
        if not queries:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No queries provided'})
            }
        
        model, processor, label_encoder = load_model()
        
        results = []
        total_start_time = time.time()
        
        for query in queries:
            intent, confidence, inference_time = predict_intent(
                query, model, processor, label_encoder
            )
            
            response_data = generate_response(intent, confidence)
            
            results.append({
                'query': query,
                'intent': intent,
                'confidence': round(confidence, 3),
                'response': response_data['message'],
                'processing_time_ms': round(inference_time * 1000, 2)
            })
        
        total_processing_time = time.time() - total_start_time
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'results': results,
                'total_queries': len(queries),
                'total_processing_time_ms': round(total_processing_time * 1000, 2),
                'average_processing_time_ms': round((total_processing_time / len(queries)) * 1000, 2),
                'timestamp': int(time.time())
            })
        }
        
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': 'Batch processing failed',
                'message': str(e),
                'timestamp': int(time.time())
            })
        }