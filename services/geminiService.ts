import { GoogleGenAI, Type } from "@google/genai";
import { AttachedImage } from "../types";

const SYSTEM_INSTRUCTION = `
You are a World-Class Prompt Engineer and Product Architect. 
Your goal is to take the user's raw, unstructured request (which may include images) and refine it into a highly structured, professional AI prompt.

**Guidelines:**
1. **Analyze:** First, understand the user's intent, audience, and core requirement. If images are provided, analyze them to extract visual style, layout, or data to include in the context.
2. **Structure:** Use the "CO-STAR" framework (Context, Objective, Style, Tone, Audience, Response) or a similarly robust structure to organize the prompt.
3. **Language:** **STRICT RULE**: The output language must strictly match the user's input language. 
   - If the user provides input in **Simplified Chinese**, the entire response (Analysis AND Refined Prompt) MUST be in **Simplified Chinese**.
   - If the user provides input in **English**, the entire response MUST be in **English**.
   - Do not translate the user's core intent into a different language unless explicitly asked.
4. **Output Format:** You must return the response in a specific layout:
   - **Analysis:** A brief explanation of what was improved (2-3 sentences).
   - **Refined Prompt:** The actual code block containing the optimized prompt.

**Role:** You are helpful, precise, and technical.
`;

export class GeminiService {
  private ai: GoogleGenAI;

  constructor() {
    // API key is injected via process.env.API_KEY
    this.ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  }

  /**
   * Refines the prompt using Gemini 3 Flash
   */
  async refinePromptStream(
    inputText: string, 
    images: AttachedImage[], 
    onChunk: (text: string) => void
  ): Promise<void> {
    
    const model = 'gemini-3-flash-preview';

    const parts: any[] = [];

    // Add images if available
    if (images.length > 0) {
      images.forEach(img => {
        // Strip data:image/xyz;base64, prefix if present for the API call
        const base64Data = img.data.split(',')[1];
        parts.push({
          inlineData: {
            mimeType: img.mimeType,
            data: base64Data
          }
        });
      });
    }

    // Add text prompt
    parts.push({
      text: `User Request: "${inputText}". \n\nPlease refine this into a professional prompt.`
    });

    try {
      const responseStream = await this.ai.models.generateContentStream({
        model: model,
        contents: {
          role: 'user',
          parts: parts
        },
        config: {
          systemInstruction: SYSTEM_INSTRUCTION,
          temperature: 0.7,
        }
      });

      for await (const chunk of responseStream) {
        if (chunk.text) {
          onChunk(chunk.text);
        }
      }
    } catch (error) {
      console.error("Gemini API Error:", error);
      throw error;
    }
  }
}

export const geminiService = new GeminiService();