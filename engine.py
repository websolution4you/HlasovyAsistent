"""
Gemma 4 Engine - Lokálne spracovanie audio-to-audio pomocou Gemma 4 (E4B/E2B).
Využíva vLLM alebo google-ai-edge SDK pre natívne audio spracovanie.
"""

import os
import asyncio
import json
from typing import Optional, AsyncGenerator, Callable
from dataclasses import dataclass
import numpy as np

# Pokus o import vLLM (preferovaná metóda pre produkciu)
try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("[engine] vLLM nie je dostupný, skúšam google-ai-edge SDK")

# Fallback na google-ai-edge SDK
try:
    import google.ai.edge as edge
    EDGE_SDK_AVAILABLE = True
except ImportError:
    EDGE_SDK_AVAILABLE = False
    print("[engine] google-ai-edge SDK nie je dostupný")


@dataclass
class AudioChunk:
    """Reprezentuje chunk audio dát."""
    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16-bit = 2 bytes


@dataclass
class ModelResponse:
    """Odpoveď z modelu."""
    audio_data: Optional[bytes] = None
    text: Optional[str] = None
    tool_call: Optional[dict] = None
    is_complete: bool = False


class Gemma4Engine:
    """
    Engine pre Gemma 4 model s natívnym audio-to-audio spracovaním.
    Minimalizuje latenciu pomocou streamovania a lokálneho GPU.
    """
    
    def __init__(
        self,
        model_path: str,
        system_prompt: str,
        tools: list = None,
        use_vllm: bool = True,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
    ):
        """
        Inicializuje Gemma 4 engine.
        
        Args:
            model_path: Cesta k lokálnemu Gemma 4 modelu (napr. "gemma-4-e4b")
            system_prompt: Systémový prompt pre model
            tools: Zoznam dostupných nástrojov (function calling)
            use_vllm: Použiť vLLM (True) alebo google-ai-edge SDK (False)
            gpu_memory_utilization: Percento GPU pamäte na použitie (0.0-1.0)
            max_model_len: Maximálna dĺžka kontextu
        """
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        
        self.engine = None
        self.is_initialized = False
        self.is_generating = False
        self.should_stop = False
        
        # Audio buffer pre streamovanie
        self.audio_buffer = bytearray()
        self.conversation_history = []
        
    async def initialize(self):
        """Inicializuje model engine."""
        if self.is_initialized:
            return
        
        print(f"[engine] Inicializujem Gemma 4 model: {self.model_path}")
        
        if self.use_vllm:
            await self._initialize_vllm()
        else:
            await self._initialize_edge_sdk()
        
        self.is_initialized = True
        print("[engine] Model úspešne inicializovaný")
    
    async def _initialize_vllm(self):
        """Inicializuje vLLM engine."""
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM nie je nainštalovaný. Nainštalujte: pip install vllm")
        
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
            dtype="auto",
            # Audio-specific konfigurácia
            enable_audio=True,
            audio_sample_rate=16000,
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("[engine] vLLM engine inicializovaný")
    
    async def _initialize_edge_sdk(self):
        """Inicializuje google-ai-edge SDK."""
        if not EDGE_SDK_AVAILABLE:
            raise RuntimeError("google-ai-edge SDK nie je nainštalovaný")
        
        # Načítaj model cez Edge SDK
        self.engine = await edge.load_model(
            self.model_path,
            device="cuda",  # Použiť GPU
            audio_enabled=True,
        )
        print("[engine] google-ai-edge SDK inicializovaný")
    
    async def process_audio_stream(
        self,
        audio_generator: AsyncGenerator[AudioChunk, None],
        tool_handler: Optional[Callable] = None,
    ) -> AsyncGenerator[ModelResponse, None]:
        """
        Spracováva streamované audio a generuje odpovede.
        
        Args:
            audio_generator: Async generátor audio chunkov od používateľa
            tool_handler: Callback funkcia pre spracovanie tool calls
        
        Yields:
            ModelResponse objekty s audio/text/tool_call dátami
        """
        if not self.is_initialized:
            await self.initialize()
        
        self.is_generating = True
        self.should_stop = False
        
        try:
            # Zbieraj audio chunks do bufferu
            audio_chunks = []
            async for chunk in audio_generator:
                if self.should_stop:
                    break
                audio_chunks.append(chunk.data)
            
            # Skonvertuj na numpy array pre model
            audio_data = b"".join(audio_chunks)
            
            # Spracuj cez model
            if self.use_vllm:
                async for response in self._process_vllm(audio_data, tool_handler):
                    yield response
            else:
                async for response in self._process_edge_sdk(audio_data, tool_handler):
                    yield response
        
        finally:
            self.is_generating = False
    
    async def _process_vllm(
        self,
        audio_data: bytes,
        tool_handler: Optional[Callable] = None,
    ) -> AsyncGenerator[ModelResponse, None]:
        """Spracuje audio cez vLLM engine."""
        # Priprav prompt s audio dátami
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history,
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": audio_data,
                        "format": "pcm16",
                        "sample_rate": 16000,
                    }
                ],
            },
        ]
        
        # Sampling parametre pre nízku latenciu
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            # Audio-specific
            audio_output=True,
            stream=True,
        )
        
        # Generuj odpoveď
        request_id = f"req_{id(audio_data)}"
        results_generator = self.engine.generate(
            prompt=messages,
            sampling_params=sampling_params,
            request_id=request_id,
        )
        
        async for result in results_generator:
            if self.should_stop:
                await self.engine.abort(request_id)
                break
            
            # Extrahuj audio a text z výstupu
            output = result.outputs[0]
            
            # Audio chunk
            if hasattr(output, "audio_data") and output.audio_data:
                yield ModelResponse(audio_data=output.audio_data)
            
            # Text (pre debugging/logging)
            if hasattr(output, "text") and output.text:
                yield ModelResponse(text=output.text)
            
            # Tool call
            if hasattr(output, "tool_calls") and output.tool_calls:
                for tool_call in output.tool_calls:
                    tool_result = None
                    if tool_handler:
                        tool_result = await tool_handler(
                            tool_call["name"],
                            tool_call.get("arguments", {}),
                        )
                    
                    yield ModelResponse(
                        tool_call={
                            "name": tool_call["name"],
                            "arguments": tool_call.get("arguments", {}),
                            "result": tool_result,
                        }
                    )
            
            # Koniec generovania
            if result.finished:
                yield ModelResponse(is_complete=True)
    
    async def _process_edge_sdk(
        self,
        audio_data: bytes,
        tool_handler: Optional[Callable] = None,
    ) -> AsyncGenerator[ModelResponse, None]:
        """Spracuje audio cez google-ai-edge SDK."""
        # Priprav input
        input_data = {
            "audio": audio_data,
            "system_prompt": self.system_prompt,
            "conversation_history": self.conversation_history,
        }
        
        # Generuj odpoveď
        async for chunk in self.engine.generate_stream(input_data):
            if self.should_stop:
                break
            
            # Audio output
            if "audio" in chunk:
                yield ModelResponse(audio_data=chunk["audio"])
            
            # Text output
            if "text" in chunk:
                yield ModelResponse(text=chunk["text"])
            
            # Tool call
            if "tool_call" in chunk:
                tool_call = chunk["tool_call"]
                tool_result = None
                if tool_handler:
                    tool_result = await tool_handler(
                        tool_call["name"],
                        tool_call.get("arguments", {}),
                    )
                
                yield ModelResponse(
                    tool_call={
                        "name": tool_call["name"],
                        "arguments": tool_call.get("arguments", {}),
                        "result": tool_result,
                    }
                )
            
            # Koniec
            if chunk.get("done", False):
                yield ModelResponse(is_complete=True)
    
    def interrupt(self):
        """
        Prerušenie (barge-in) - okamžite zastaví generovanie odpovede.
        Volá sa keď používateľ začne hovoriť počas odpovede modelu.
        """
        print("[engine] Barge-in detekovaný - zastavujem generovanie")
        self.should_stop = True
    
    def add_to_history(self, role: str, content: str):
        """Pridá správu do konverzačnej histórie."""
        self.conversation_history.append({
            "role": role,
            "content": content,
        })
    
    def clear_history(self):
        """Vymaže konverzačnú históriu."""
        self.conversation_history.clear()
    
    async def cleanup(self):
        """Uvoľní zdroje."""
        if self.engine:
            if self.use_vllm and hasattr(self.engine, "shutdown"):
                await self.engine.shutdown()
            self.engine = None
        self.is_initialized = False
        print("[engine] Engine cleanup dokončený")


# Pomocné funkcie pre detekciu reči (Voice Activity Detection)

def detect_speech(audio_chunk: bytes, threshold: float = 0.02) -> bool:
    """
    Jednoduchá detekcia reči na základe energie signálu.
    
    Args:
        audio_chunk: PCM 16-bit audio dáta
        threshold: Prah energie pre detekciu reči
    
    Returns:
        True ak je detekovaná reč, inak False
    """
    if len(audio_chunk) < 2:
        return False
    
    # Konvertuj na numpy array
    audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
    
    # Vypočítaj RMS (Root Mean Square) energiu
    rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
    
    # Normalizuj na rozsah 0-1
    normalized_rms = rms / 32768.0
    
    return normalized_rms > threshold


def calculate_audio_duration(audio_bytes: bytes, sample_rate: int = 16000) -> float:
    """
    Vypočíta dĺžku audio v sekundách.
    
    Args:
        audio_bytes: PCM 16-bit audio dáta
        sample_rate: Vzorkovacia frekvencia
    
    Returns:
        Dĺžka v sekundách
    """
    num_samples = len(audio_bytes) // 2  # 16-bit = 2 bytes per sample
    return num_samples / sample_rate
