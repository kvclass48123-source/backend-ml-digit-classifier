import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '1mb' }));

  // API Route for Prediction
  // NOTE: This is a fallback implementation for the AI Studio preview.
  // The actual implementation requested by the user is in app.py.
  app.post("/api/predict", async (req, res) => {
    try {
      const { data } = req.body;
      if (!data || !Array.isArray(data)) {
        return res.status(400).json({ error: "Invalid data format" });
      }

      // Check for Gemini API key
      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey || apiKey === "MY_GEMINI_API_KEY") {
        // Mock prediction if no API key
        // Send a random digit for visual demonstration if Gemini is not available
        const mockDigit = Math.floor(Math.random() * 10);
        return res.json({
          prediction: mockDigit.toString(),
          confidence: 0.95,
          note: "Demo mode (Mock)"
        });
      }

      // Use Gemini to classify the digit for the preview
      const genAI = new GoogleGenAI(apiKey);
      const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

      // Represent data as a string (visual grid) to help Gemini
      // Since it's 28x28, we can show it as a series of points or just tell it the pixels.
      // A 28x28 grid of ASCII might work too.
      let gridStr = "";
      for (let i = 0; i < 28; i++) {
        for (let j = 0; j < 28; j++) {
          const val = data[i * 28 + j];
          gridStr += val > 0.5 ? "@" : ".";
        }
        gridStr += "\n";
      }

      const prompt = `You are an MNIST digit classifier. Here is a 28x28 grayscale image represented as an ASCII grid (@ is ink, . is paper):
      
${gridStr}

Identify the handwritten digit (0-9). Return ONLY the digit followed by a comma and a confidence score between 0 and 1.
Example: 5, 0.98`;

      const result = await model.generateContent(prompt);
      const responseText = result.response.text().trim();
      
      const [digit, confidence] = responseText.split(",").map(s => s.trim());
      
      res.json({
        prediction: digit || "0",
        confidence: parseFloat(confidence) || 0.9,
        note: "Classified via Gemini Preview Fallback"
      });
    } catch (error) {
      console.error("Prediction error:", error);
      res.status(500).json({ error: "Failed to process prediction" });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
