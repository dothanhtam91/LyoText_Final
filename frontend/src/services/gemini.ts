/**
 * TTS service (client-side).
 * Uses Web Speech API by default (reliable, no autoplay blocks). Falls back to Gemini TTS when configured.
 */

import { GoogleGenAI, Modality } from "@google/genai";

const hasGeminiKey = !!process.env.GEMINI_API_KEY;
const ai = hasGeminiKey ? new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY }) : null;

/** Speak text via browser Web Speech API — works without API key, no autoplay restrictions. */
function speakTextWebSpeech(text: string): Promise<void> {
  return new Promise((resolve) => {
    if (!('speechSynthesis' in window)) {
      console.warn("Web Speech API not available");
      resolve();
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.volume = 1;
    utterance.onend = () => resolve();
    utterance.onerror = (e) => {
      console.warn("Speech synthesis error:", e);
      resolve();
    };
    window.speechSynthesis.speak(utterance);
  });
}

/** Speak text via Gemini TTS. Returns data URL for audio, or null on failure. */
async function speakTextGemini(text: string): Promise<string | null> {
  if (!ai) return null;
  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-preview-tts",
      contents: [{ parts: [{ text }] }],
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: 'Kore' },
          },
        },
      },
    });
    const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (base64Audio) return `data:audio/mp3;base64,${base64Audio}`;
    return null;
  } catch (error) {
    console.warn("Gemini TTS failed, using Web Speech:", error);
    return null;
  }
}

/**
 * Speak text out loud using Web Speech API (built-in, no autoplay blocks).
 * Exported for fallback when Audio.play() is blocked.
 */
export async function speakWithWebSpeech(text: string): Promise<void> {
  if (!text?.trim()) return;
  await speakTextWebSpeech(text);
}

/**
 * Speak text out loud.
 * Uses Web Speech API by default (reliable, no autoplay). Uses Gemini when configured and working.
 */
export async function speakText(text: string): Promise<string | null> {
  if (!text?.trim()) return null;
  if (hasGeminiKey) {
    const audioDataUrl = await speakTextGemini(text);
    if (audioDataUrl) return audioDataUrl;
  }
  await speakTextWebSpeech(text);
  return null;
}
