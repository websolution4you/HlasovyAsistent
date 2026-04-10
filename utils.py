"""
Audio conversion utilities for Twilio Media Streams.
Handles conversion between Twilio's mu-law 8kHz format and PCM 16-bit format.
"""

try:
    import audioop
except ImportError:
    try:
        import audioop_lts as audioop  # type: ignore
    except ImportError:
        audioop = None  # type: ignore


def mulaw_to_pcm16(mulaw_bytes: bytes, target_rate: int = 16000) -> bytes:
    """
    Konvertuje mu-law 8kHz (Twilio) na PCM 16-bit s cieľovou vzorkovacou frekvenciou.
    
    Args:
        mulaw_bytes: Surové mu-law audio dáta (8kHz)
        target_rate: Cieľová vzorkovacia frekvencia (default: 16000 Hz)
    
    Returns:
        PCM 16-bit audio dáta v cieľovej vzorkovacej frekvencii
    """
    if audioop is None:
        # Núdzový fallback bez audioop (nie je ideálne)
        return mulaw_bytes * (target_rate // 8000)
    
    # mu-law -> PCM 16-bit @ 8kHz
    pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
    
    # Resample z 8kHz na target_rate
    if target_rate != 8000:
        pcm_target, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, target_rate, None)
        return pcm_target
    
    return pcm_8k


def pcm16_to_mulaw(pcm_bytes: bytes, source_rate: int = 16000) -> bytes:
    """
    Konvertuje PCM 16-bit na mu-law 8kHz (pre Twilio).
    
    Args:
        pcm_bytes: PCM 16-bit audio dáta
        source_rate: Vzorkovacia frekvencia vstupných dát (default: 16000 Hz)
    
    Returns:
        mu-law 8kHz audio dáta pre Twilio
    """
    if audioop is None:
        # Núdzový fallback
        return pcm_bytes[: len(pcm_bytes) // (source_rate // 8000)]
    
    # Resample na 8kHz ak je potrebné
    if source_rate != 8000:
        pcm_8k, _ = audioop.ratecv(pcm_bytes, 2, 1, source_rate, 8000, None)
    else:
        pcm_8k = pcm_bytes
    
    # PCM -> mu-law
    return audioop.lin2ulaw(pcm_8k, 2)


def pcm16_to_pcm24(pcm16_bytes: bytes) -> bytes:
    """
    Konvertuje PCM 16-bit na PCM 24-bit (pre niektoré modely).
    
    Args:
        pcm16_bytes: PCM 16-bit audio dáta
    
    Returns:
        PCM 24-bit audio dáta
    """
    if audioop is None:
        # Jednoduchý fallback - pridaj nulový byte
        result = bytearray()
        for i in range(0, len(pcm16_bytes), 2):
            result.extend(pcm16_bytes[i:i+2])
            result.append(0)
        return bytes(result)
    
    # Konverzia cez audioop
    return audioop.lin2lin(pcm16_bytes, 2, 3)


def normalize_audio_level(pcm_bytes: bytes, target_level: float = 0.8) -> bytes:
    """
    Normalizuje úroveň audia (voliteľné - pre lepšiu kvalitu).
    
    Args:
        pcm_bytes: PCM 16-bit audio dáta
        target_level: Cieľová úroveň (0.0 - 1.0)
    
    Returns:
        Normalizované PCM audio dáta
    """
    if audioop is None or len(pcm_bytes) == 0:
        return pcm_bytes
    
    # Zisti aktuálnu úroveň
    max_val = audioop.max(pcm_bytes, 2)
    if max_val == 0:
        return pcm_bytes
    
    # Vypočítaj faktor zosilnenia
    target_max = int(32767 * target_level)
    factor = target_max / max_val
    
    # Aplikuj zosilnenie
    return audioop.mul(pcm_bytes, 2, factor)
