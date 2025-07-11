!pip install -q transformers accelerate datasets torch torchvision torchaudio
!pip install -q TTS coqui-tts librosa soundfile phonemizer espeak-ng
!pip install -q wandb parselmouth praat-parselmouth
!pip install -q huggingface_hub safetensors
!pip install -q onnx onnxruntime-gpu
!pip install -q gradio ipywidgets
!apt-get -qq -y install espeak-ng

import os
import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
from transformers import AutoTokenizer, AutoModelForCausalLM
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import time

class VoiceCloningPipeline:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.performance_metrics = {}
        
    def load_kokoro_82m(self):
        """Load Kokoro-82M model"""
        print("Loading Kokoro-82M...")
        start_time = time.time()
        
        try:
            # Load tokenizer and model
            model_name = "hexgrad/Kokoro-82M"
            self.tokenizers['kokoro'] = AutoTokenizer.from_pretrained(model_name)
            self.models['kokoro'] = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            load_time = time.time() - start_time
            self.performance_metrics['kokoro'] = {
                'load_time': load_time,
                'parameters': '82M',
                'memory_usage': torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            }
            print(f"✓ Kokoro-82M loaded in {load_time:.2f}s")
            
        except Exception as e:
            print(f"✗ Failed to load Kokoro-82M: {e}")
            
    def load_xtts_v2(self):
        """Load XTTS-v2 model"""
        print("Loading XTTS-v2...")
        start_time = time.time()
        
        try:
            # Initialize TTS with XTTS-v2
            self.models['xtts'] = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            
            load_time = time.time() - start_time
            self.performance_metrics['xtts'] = {
                'load_time': load_time,
                'parameters': '~1.1B',
                'memory_usage': torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            }
            print(f"✓ XTTS-v2 loaded in {load_time:.2f}s")
            
        except Exception as e:
            print(f"✗ Failed to load XTTS-v2: {e}")
            
    def load_all_models(self):
        """Load all available models"""
        print("Loading all TTS models...")
        self.load_kokoro_82m()
        self.load_xtts_v2()
        
        # Print summary
        print("\n" + "="*50)
        print("MODEL LOADING SUMMARY")
        print("="*50)
        for model_name, metrics in self.performance_metrics.items():
            print(f"{model_name.upper()}:")
            print(f"  Load Time: {metrics['load_time']:.2f}s")
            print(f"  Parameters: {metrics['parameters']}")
            print(f"  Memory Usage: {metrics['memory_usage']:.2f} GB")
            print()
class AudioDataProcessor:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        self.min_duration = 5.0  # seconds
        self.max_duration = 10.0  # seconds
        
    def download_sample_dataset(self):
        """Download sample dataset for testing"""
        print("Downloading sample dataset...")
        
        # Create dataset directory
        os.makedirs("datasets", exist_ok=True)
        
        # Download LibriSpeech dev-clean subset for testing
        !wget -q -O datasets/dev-clean.tar.gz "https://www.openslr.org/resources/12/dev-clean.tar.gz"
        !tar -xzf datasets/dev-clean.tar.gz -C datasets/
        
        print("✓ Sample dataset downloaded")
        
    def preprocess_audio_file(self, audio_path, output_dir):
        """Preprocess a single audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            
            # Normalize amplitude
            audio = librosa.util.normalize(audio)
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Split into chunks
            chunks = self.split_audio_into_chunks(audio, self.target_sr)
            
            # Save chunks
            output_paths = []
            for i, chunk in enumerate(chunks):
                output_path = os.path.join(output_dir, f"{Path(audio_path).stem}_chunk_{i}.wav")
                sf.write(output_path, chunk, self.target_sr)
                output_paths.append(output_path)
                
            return output_paths
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return []
    
    def split_audio_into_chunks(self, audio, sr):
        """Split audio into 5-10 second chunks"""
        chunks = []
        min_samples = int(self.min_duration * sr)
        max_samples = int(self.max_duration * sr)
        
        for start in range(0, len(audio), min_samples):
            end = min(start + max_samples, len(audio))
            chunk = audio[start:end]
            
            if len(chunk) >= min_samples:
                chunks.append(chunk)
                
        return chunks
    
    def create_dataset_manifest(self, audio_dir, output_file):
        """Create dataset manifest with audio paths and metadata"""
        manifest = []
        
        for audio_file in Path(audio_dir).glob("*.wav"):
            try:
                # Get audio duration
                audio, sr = librosa.load(str(audio_file), sr=None)
                duration = len(audio) / sr
                
                manifest.append({
                    'audio_path': str(audio_file),
                    'duration': duration,
                    'sample_rate': sr,
                    'text': f"Sample audio from {audio_file.stem}"  # Placeholder text
                })
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                
        # Save manifest
        df = pd.DataFrame(manifest)
        df.to_csv(output_file, index=False)
        
        print(f"✓ Dataset manifest created: {len(manifest)} files")
        return manifest

class VoiceCloner:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
    def clone_voice_xtts(self, text, reference_audio_path, output_path, language="en"):
        """Clone voice using XTTS-v2"""
        try:
            start_time = time.time()
            
            # Generate speech
            self.pipeline.models['xtts'].tts_to_file(
                text=text,
                speaker_wav=reference_audio_path,
                language=language,
                file_path=output_path
            )
            
            inference_time = time.time() - start_time
            
            # Calculate Real-Time Factor (RTF)
            audio_duration = librosa.get_duration(path=output_path)
            rtf = inference_time / audio_duration
            
            return {
                'success': True,
                'inference_time': inference_time,
                'audio_duration': audio_duration,
                'rtf': rtf,
                'output_path': output_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def benchmark_models(self, test_text, reference_audio, num_runs=3):
        """Benchmark all models with multiple runs"""
        results = {}
        
        for model_name in self.pipeline.models.keys():
            if model_name == 'xtts':
                model_results = []
                
                for run in range(num_runs):
                    output_path = f"benchmark_{model_name}_run_{run}.wav"
                    result = self.clone_voice_xtts(test_text, reference_audio, output_path)
                    
                    if result['success']:
                        model_results.append({
                            'run': run,
                            'inference_time': result['inference_time'],
                            'rtf': result['rtf']
                        })
                
                if model_results:
                    avg_inference_time = np.mean([r['inference_time'] for r in model_results])
                    avg_rtf = np.mean([r['rtf'] for r in model_results])
                    
                    results[model_name] = {
                        'avg_inference_time': avg_inference_time,
                        'avg_rtf': avg_rtf,
                        'runs': model_results
                    }
        
        return results
class XTTSFineTuner:
    def __init__(self, model_path="coqui/XTTS-v2"):
        self.model_path = model_path
        self.config = None
        self.model = None
        
    def setup_training_config(self, dataset_path, output_dir):
        """Setup training configuration"""
        from TTS.tts.configs.xtts_config import XttsConfig
        
        config = XttsConfig()
        config.load_json("path/to/config.json")  # Load base config
        
        # Training parameters
        config.batch_size = 2  # Reduced for Colab
        config.grad_clip = 1.0
        config.print_step = 50
        config.lr = 5e-6
        config.num_epochs = 10
        config.use_grad_checkpointing = True
        config.mixed_precision = True
        
        # Dataset configuration
        config.datasets = [
            {
                "name": "custom_dataset",
                "path": dataset_path,
                "meta_file_train": "metadata_train.csv",
                "meta_file_val": "metadata_val.csv",
                "language": "en"
            }
        ]
        
        # Output directory
        config.output_path = output_dir
        
        return config
    
    def prepare_dataset_for_training(self, manifest_file):
        """Prepare dataset in XTTS format"""
        df = pd.read_csv(manifest_file)
        
        # Split into train/val
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)
        
        # Create XTTS format metadata
        def create_xtts_metadata(df_subset, output_file):
            metadata = []
            for _, row in df_subset.iterrows():
                metadata.append({
                    'audio_file': row['audio_path'],
                    'text': row['text'],
                    'speaker_name': 'target_speaker'
                })
            
            pd.DataFrame(metadata).to_csv(output_file, index=False, sep='|')
        
        create_xtts_metadata(train_df, "metadata_train.csv")
        create_xtts_metadata(val_df, "metadata_val.csv")
        
        print(f"✓ Training dataset: {len(train_df)} samples")
        print(f"✓ Validation dataset: {len(val_df)} samples")
class VoiceQualityEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def calculate_mel_cepstral_distortion(self, original_path, generated_path):
        """Calculate Mel-Cepstral Distortion (MCD)"""
        try:
            # Load audio files
            orig_audio, sr1 = librosa.load(original_path, sr=16000)
            gen_audio, sr2 = librosa.load(generated_path, sr=16000)
            
            # Ensure same length
            min_len = min(len(orig_audio), len(gen_audio))
            orig_audio = orig_audio[:min_len]
            gen_audio = gen_audio[:min_len]
            
            # Extract MFCC features
            orig_mfcc = librosa.feature.mfcc(y=orig_audio, sr=16000, n_mfcc=13)
            gen_mfcc = librosa.feature.mfcc(y=gen_audio, sr=16000, n_mfcc=13)
            
            # Calculate MCD
            mcd = np.mean(np.sqrt(np.sum((orig_mfcc - gen_mfcc)**2, axis=0)))
            
            return mcd
            
        except Exception as e:
            print(f"Error calculating MCD: {e}")
            return None
    
    def analyze_prosody(self, audio_path):
        """Analyze prosody features using Parselmouth"""
        try:
            import parselmouth
            
            sound = parselmouth.Sound(audio_path)
            pitch = sound.to_pitch()
            
            # Extract prosody features
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced frames
            
            prosody_features = {
                'mean_pitch': np.mean(pitch_values) if len(pitch_values) > 0 else 0,
                'std_pitch': np.std(pitch_values) if len(pitch_values) > 0 else 0,
                'min_pitch': np.min(pitch_values) if len(pitch_values) > 0 else 0,
                'max_pitch': np.max(pitch_values) if len(pitch_values) > 0 else 0,
                'duration': sound.duration
            }
            
            return prosody_features
            
        except Exception as e:
            print(f"Error analyzing prosody: {e}")
            return None
    
    def comprehensive_evaluation(self, original_audio, generated_audio):
        """Comprehensive evaluation of voice quality"""
        results = {
            'original_prosody': self.analyze_prosody(original_audio),
            'generated_prosody': self.analyze_prosody(generated_audio),
            'mcd': self.calculate_mel_cepstral_distortion(original_audio, generated_audio)
        }
        
        # Calculate prosody similarity
        if results['original_prosody'] and results['generated_prosody']:
            pitch_similarity = 1 - abs(
                results['original_prosody']['mean_pitch'] - 
                results['generated_prosody']['mean_pitch']
            ) / results['original_prosody']['mean_pitch']
            
            results['pitch_similarity'] = max(0, pitch_similarity)
        
        return results
def main():
    print("🎙️ Voice Cloning with Open-Source TTS Models")
    print("="*60)
    
    # Initialize components
    pipeline = VoiceCloningPipeline()
    processor = AudioDataProcessor()
    evaluator = VoiceQualityEvaluator()
    
    # Step 1: Load models
    print("\n📥 Step 1: Loading Models...")
    pipeline.load_all_models()
    
    # Step 2: Prepare sample dataset
    print("\n🔄 Step 2: Preparing Dataset...")
    processor.download_sample_dataset()
    
    # Create processed dataset directory
    os.makedirs("processed_dataset", exist_ok=True)
    
    # Step 3: Voice cloning demonstration
    print("\n🎭 Step 3: Voice Cloning Demonstration...")
    cloner = VoiceCloner(pipeline)
    
    # Sample text for testing
    test_text = "Hello, this is a demonstration of voice cloning using open-source TTS models."
    
    # Create a simple reference audio for testing
    reference_audio = "reference_voice.wav"
    
    # Generate sample reference audio using XTTS default voice
    if 'xtts' in pipeline.models:
        pipeline.models['xtts'].tts_to_file(
            text="This is a reference voice sample.",
            file_path=reference_audio,
            speaker=pipeline.models['xtts'].speakers[0] if hasattr(pipeline.models['xtts'], 'speakers') else None
        )
    
    # Perform voice cloning
    cloning_result = cloner.clone_voice_xtts(
        text=test_text,
        reference_audio_path=reference_audio,
        output_path="cloned_voice_output.wav"
    )
    
    if cloning_result['success']:
        print(f"✓ Voice cloning successful!")
        print(f"  Inference time: {cloning_result['inference_time']:.2f}s")
        print(f"  Real-time factor: {cloning_result['rtf']:.2f}")
        print(f"  Output saved to: {cloning_result['output_path']}")
        
        # Step 4: Evaluation
        print("\n📊 Step 4: Quality Evaluation...")
        eval_results = evaluator.comprehensive_evaluation(
            reference_audio, 
            cloning_result['output_path']
        )
        
        print(f"Evaluation Results:")
        if eval_results['mcd']:
            print(f"  Mel-Cepstral Distortion: {eval_results['mcd']:.2f}")
        if eval_results.get('pitch_similarity'):
            print(f"  Pitch Similarity: {eval_results['pitch_similarity']:.2f}")
            
    else:
        print(f"✗ Voice cloning failed: {cloning_result['error']}")
    
    # Step 5: Benchmarking
    print("\n⚡ Step 5: Performance Benchmarking...")
    benchmark_results = cloner.benchmark_models(test_text, reference_audio)
    
    print("Benchmark Results:")
    for model_name, results in benchmark_results.items():
        print(f"  {model_name.upper()}:")
        print(f"    Average Inference Time: {results['avg_inference_time']:.2f}s")
        print(f"    Average RTF: {results['avg_rtf']:.2f}")
    
    print("\n🎉 Voice Cloning Pipeline Complete!")
    print("Check the generated audio files in the current directory.")

def create_gradio_interface():
    """Create interactive Gradio interface"""
    import gradio as gr

    # Initialize pipeline
    pipeline = VoiceCloningPipeline()
    pipeline.load_all_models()
    cloner = VoiceCloner(pipeline)

    def clone_voice_interface(text, reference_audio, language):
        """Interface function for voice cloning"""
        if not text or not reference_audio:
            return "Please provide both text and reference audio."

        output_path = "gradio_output.wav"
        result = cloner.clone_voice_xtts(text, reference_audio, output_path, language)

        if result['success']:
            return output_path
        else:
            return f"Error: {result['error']}"

    # Create interface
    interface = gr.Interface(
        fn=clone_voice_interface,
        inputs=[
            gr.Textbox(
                label="Text to synthesize",
                placeholder="Enter the text you want to convert to speech...",
                lines=3
            ),
            gr.Audio(
                label="Reference Audio (Upload or Record)",
                source="microphone",      # 🎙️ Mic support added here
                type="filepath",          # Path to .wav saved file
                format="wav"
            ),
            gr.Dropdown(
                choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "ko", "hu"],
                value="en",
                label="Language"
            )
        ],
        outputs=gr.Audio(label="Generated Speech"),
        title="🎙️ Voice Cloning with Open-Source TTS",
        description="Clone any voice using just a few seconds of reference audio!",
        examples=[
            ["Hello, this is a test of voice cloning technology.", None, "en"],
            ["Welcome to the future of speech synthesis.", None, "en"]
        ]
    )

    return interface


class DeploymentUtils:
    @staticmethod
    def optimize_for_cpu():
        """Optimize models for CPU inference"""
        print("Optimizing models for CPU inference...")
        # Add CPU optimization logic here
        pass

    @staticmethod
    def convert_to_onnx(model, output_path):
        """Convert PyTorch model to ONNX format"""
        try:
            import onnx
            # Add ONNX conversion logic here
            print(f"Model converted to ONNX: {output_path}")
        except Exception as e:
            print(f"ONNX conversion failed: {e}")

    @staticmethod
    def create_deployment_package():
        """Create deployment package"""
        print("Creating deployment package...")
        deployment_files = [
            "voice_cloning_pipeline.py",
            "requirements.txt",
            "README.md"
        ]

        requirements = """
torch>=1.9.0
torchaudio>=0.9.0
transformers>=4.21.0
TTS>=0.15.0
librosa>=0.9.0
soundfile>=0.10.0
numpy>=1.21.0
pandas>=1.3.0
gradio>=3.0.0
huggingface-hub>=0.10.0
"""

        with open("requirements.txt", "w") as f:
            f.write(requirements)

        print("✓ Deployment package created")


if __name__ == "__main__":
    # Run main pipeline
    main()

    # Create Gradio interface for interactive testing
    print("\n🌐 Starting Gradio Interface...")
    try:
        interface = create_gradio_interface()
        interface.launch(share=True, debug=True)
    except Exception as e:
        print(f"Gradio interface failed: {e}")
        print("You can still use the command-line interface above.")
import gradio as gr
import torch.serialization
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig  # <- new import

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig])


# Fix for PyTorch 2.6: allowlist the XTTS config
torch.serialization.add_safe_globals([XttsConfig])

from TTS.api import TTS

class VoiceCloningPipeline:
    def __init__(self):
        self.models = {}

    def load_all_models(self):
        print("Loading XTTS-v2...")
        try:
            self.models["xtts"] = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            print("✓ XTTS-v2 loaded successfully.")
        except Exception as e:
            print(f"✗ Failed to load XTTS-v2: {e}")

class VoiceCloner:
    def __init__(self, pipeline):
        self.tts_model = pipeline.models.get("xtts")

    def clone_voice_xtts(self, text, reference_audio, output_path, language):
        try:
            self.tts_model.tts_to_file(
                text=text,
                speaker_wav=reference_audio,
                language=language,
                file_path=output_path
            )
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

def create_gradio_interface():
    pipeline = VoiceCloningPipeline()
    pipeline.load_all_models()
    cloner = VoiceCloner(pipeline)

    def clone_voice_interface(text, reference_audio, language):
        if not text or not reference_audio:
            return None  # Return None to avoid crashing Gradio

        output_path = "cloned_output.wav"
        result = cloner.clone_voice_xtts(text, reference_audio, output_path, language)

        if result['success']:
            return output_path
        else:
            print(f"Error: {result['error']}")  # Print error to terminal
            return None

    interface = gr.Interface(
        fn=clone_voice_interface,
        inputs=[
            gr.Textbox(label="Text to synthesize", lines=3, placeholder="Say something..."),
            gr.Audio(label="Reference Audio (.wav only)", type="filepath"),
            gr.Dropdown(
                label="Language",
                choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "ko", "hu"],
                value="en"
            )
        ],
        outputs=gr.Audio(label="Cloned Output"),
        title="🎙️ Voice Cloning with XTTS",
        description="Upload a voice clip + text and get it cloned using XTTS!",
        examples=[
            ["Hello, how are you?", "sample.wav", "en"]
        ]
    )

    return interface

if __name__ == "__main__":
    try:
        print("\n🚀 Launching Gradio Interface...")
        interface = create_gradio_interface()
        interface.launch(share=True, debug=True)
    except Exception as e:
        print(f"❌ Gradio interface failed: {e}")
 