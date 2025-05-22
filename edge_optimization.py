import torch
import torch.nn as nn
import torch.quantization as quantization
import torch.jit as jit
import onnx
import onnxruntime as ort
import numpy as np
import time
from collections import OrderedDict
import json
import os

class OptimizedIntentClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=3, 
                 n_classes=20, max_len=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
    def forward(self, x):
        seq_len = x.size(1)
        
        x = self.embedding(x) * (self.d_model ** 0.5)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        padding_mask = (x.sum(dim=-1) == 0)
        
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        x = x.mean(dim=1)
        return self.classifier(x)

class ModelOptimizer:
    def __init__(self, model, processor, label_encoder):
        self.model = model
        self.processor = processor
        self.label_encoder = label_encoder
        
    def quantize_model(self):
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, 
            {nn.Linear}, 
            dtype=torch.qint8
        )
        return quantized_model
    
    def prune_model(self, sparsity=0.2):
        from torch.nn.utils import prune
        
        parameters_to_prune = []
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )
        
        for module, param in parameters_to_prune:
            prune.remove(module, param)
            
        return self.model
    
    def convert_to_torchscript(self, sample_input):
        self.model.eval()
        traced_model = torch.jit.trace(self.model, sample_input)
        return traced_model
    
    def convert_to_onnx(self, sample_input, onnx_path):
        self.model.eval()
        
        torch.onnx.export(
            self.model,
            sample_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        return onnx_path
    
    def optimize_for_inference(self):
        optimized_model = OptimizedIntentClassifier(
            vocab_size=self.processor.vocab_size,
            d_model=128,
            n_heads=4,
            n_layers=3,
            n_classes=len(self.label_encoder.classes_)
        )
        
        optimized_model.load_state_dict(self.model.state_dict(), strict=False)
        
        optimized_model.eval()
        for param in optimized_model.parameters():
            param.requires_grad = False
            
        return optimized_model

class EdgeDeploymentPackager:
    def __init__(self, model, processor, label_encoder):
        self.model = model
        self.processor = processor
        self.label_encoder = label_encoder
        
    def create_deployment_package(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        optimizer = ModelOptimizer(self.model, self.processor, self.label_encoder)
        
        sample_input = torch.randint(0, self.processor.vocab_size, (1, 128))
        
        original_size = self.get_model_size(self.model)
        print(f"Original model size: {original_size:.2f} MB")
        
        quantized_model = optimizer.quantize_model()
        quantized_size = self.get_model_size(quantized_model)
        print(f"Quantized model size: {quantized_size:.2f} MB")
        
        pruned_model = optimizer.prune_model(sparsity=0.3)
        pruned_size = self.get_model_size(pruned_model)
        print(f"Pruned model size: {pruned_size:.2f} MB")
        
        traced_model = optimizer.convert_to_torchscript(sample_input)
        traced_size = self.get_model_size(traced_model)
        print(f"TorchScript model size: {traced_size:.2f} MB")
        
        onnx_path = os.path.join(output_dir, 'model.onnx')
        optimizer.convert_to_onnx(sample_input, onnx_path)
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"ONNX model size: {onnx_size:.2f} MB")
        
        torch.save({
            'quantized_model': quantized_model.state_dict(),
            'traced_model': traced_model,
            'processor': self.processor,
            'label_encoder': self.label_encoder,
            'model_config': {
                'vocab_size': self.processor.vocab_size,
                'n_classes': len(self.label_encoder.classes_),
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 3
            }
        }, os.path.join(output_dir, 'optimized_models.pth'))
        
        self.create_inference_script(output_dir)
        self.create_deployment_config(output_dir)
        
        return output_dir
    
    def get_model_size(self, model):
        if hasattr(model, 'state_dict'):
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            return (param_size + buffer_size) / (1024 * 1024)
        else:
            return os.path.getsize(model) / (1024 * 1024) if isinstance(model, str) else 0
    
    def create_inference_script(self, output_dir):
        inference_script = '''
import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import time
import json

class EdgeInference:
    def __init__(self, model_path, use_onnx=True):
        self.use_onnx = use_onnx
        
        if use_onnx:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.processor = checkpoint['processor']
            self.label_encoder = checkpoint['label_encoder']
            
            from model import OptimizedIntentClassifier
            config = checkpoint['model_config']
            
            self.model = OptimizedIntentClassifier(**config)
            self.model.load_state_dict(checkpoint['quantized_model'])
            self.model.eval()
    
    def predict(self, text):
        start_time = time.time()
        
        if self.use_onnx:
            encoded_text = self.processor.encode(text).numpy().reshape(1, -1)
            output = self.session.run(None, {self.input_name: encoded_text})[0]
            predicted_class = np.argmax(output, axis=1)[0]
            confidence = np.max(np.softmax(output, axis=1))
        else:
            with torch.no_grad():
                encoded_text = self.processor.encode(text).unsqueeze(0)
                output = self.model(encoded_text)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1).max().item()
        
        intent = self.label_encoder.inverse_transform([predicted_class])[0]
        inference_time = time.time() - start_time
        
        return {
            'intent': intent,
            'confidence': float(confidence),
            'inference_time_ms': inference_time * 1000
        }

def benchmark_models():
    texts = [
        "What are the prerequisites for CS101?",
        "I need help with my assignment",
        "When is the final exam?",
        "What is my current grade?"
    ]
    
    pytorch_inference = EdgeInference('optimized_models.pth', use_onnx=False)
    onnx_inference = EdgeInference('model.onnx', use_onnx=True)
    
    pytorch_times = []
    onnx_times = []
    
    for text in texts:
        result_pytorch = pytorch_inference.predict(text)
        result_onnx = onnx_inference.predict(text)
        
        pytorch_times.append(result_pytorch['inference_time_ms'])
        onnx_times.append(result_onnx['inference_time_ms'])
        
        print(f"Text: {text}")
        print(f"PyTorch - Intent: {result_pytorch['intent']}, Time: {result_pytorch['inference_time_ms']:.2f}ms")
        print(f"ONNX - Intent: {result_onnx['intent']}, Time: {result_onnx['inference_time_ms']:.2f}ms")
        print("-" * 80)
    
    print(f"Average PyTorch inference time: {np.mean(pytorch_times):.2f}ms")
    print(f"Average ONNX inference time: {np.mean(onnx_times):.2f}ms")
    print(f"Speedup: {np.mean(pytorch_times) / np.mean(onnx_times):.2f}x")

if __name__ == "__main__":
    benchmark_models()
'''
        
        with open(os.path.join(output_dir, 'edge_inference.py'), 'w') as f:
            f.write(inference_script)
    
    def create_deployment_config(self, output_dir):
        config = {
            'model_info': {
                'vocab_size': self.processor.vocab_size,
                'n_classes': len(self.label_encoder.classes_),
                'max_sequence_length': 128,
                'model_type': 'transformer'
            },
            'optimization': {
                'quantization': 'int8',
                'pruning_sparsity': 0.3,
                'target_latency_ms': 50,
                'memory_limit_mb': 512
            },
            'deployment': {
                'platforms': ['lambda', 'edge', 'mobile'],
                'runtime_requirements': [
                    'torch>=1.9.0',
                    'onnxruntime>=1.8.0',
                    'numpy>=1.19.0'
                ]
            },
            'performance_targets': {
                'accuracy_threshold': 0.85,
                'latency_p95_ms': 100,
                'throughput_qps': 1000,
                'memory_usage_mb': 256
            }
        }
        
        with open(os.path.join(output_dir, 'deployment_config.json'), 'w') as f:
            json.dump(config, f, indent=2)

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_latency': 0,
            'p95_latency': 0,
            'intents_processed': {},
            'confidence_distribution': []
        }
        self.latencies = []
    
    def record_request(self, intent, confidence, latency_ms, success=True):
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        self.latencies.append(latency_ms)
        self.metrics['confidence_distribution'].append(confidence)
        
        if intent in self.metrics['intents_processed']:
            self.metrics['intents_processed'][intent] += 1
        else:
            self.metrics['intents_processed'][intent] = 1
        
        self.update_latency_metrics()
    
    def update_latency_metrics(self):
        if self.latencies:
            self.metrics['average_latency'] = np.mean(self.latencies)
            self.metrics['p95_latency'] = np.percentile(self.latencies, 95)
    
    def get_metrics(self):
        return self.metrics
    
    def export_metrics(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)

def create_optimized_chatbot():
    from model import train_model
    
    print("Training base model...")
    model, processor, label_encoder = train_model()
    
    print("Creating edge deployment package...")
    packager = EdgeDeploymentPackager(model, processor, label_encoder)
    output_dir = packager.create_deployment_package('deployment_package')
    
    print(f"Deployment package created in: {output_dir}")
    
    print("Performance optimization completed!")
    print("Files created:")
    print("- optimized_models.pth (PyTorch models)")
    print("- model.onnx (ONNX model)")
    print("- edge_inference.py (Inference script)")
    print("- deployment_config.json (Configuration)")
    
    return output_dir

if __name__ == "__main__":
    create_optimized_chatbot()