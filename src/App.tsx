/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Eraser, Play, RefreshCw, Cpu, BrainCircuit, Activity } from 'lucide-react';

export default function App() {
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState<{ digit: string; confidence: number } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const contextRef = useRef<CanvasRenderingContext2D | null>(null);

  // Logical size for MNIST
  const MNIST_SIZE = 28;
  // Display size
  const DISPLAY_SIZE = 280;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = DISPLAY_SIZE;
    canvas.height = DISPLAY_SIZE;
    
    const context = canvas.getContext('2d');
    if (!context) return;

    context.lineCap = 'round';
    context.strokeStyle = 'white';
    context.lineWidth = 20; // Thick enough for the 280px canvas
    contextRef.current = context;

    // Fill background with black
    clearCanvas();
  }, []);

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const context = contextRef.current;
    if (!canvas || !context) return;

    context.fillStyle = 'black';
    context.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
    setError(null);
  };

  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    const { offsetX, offsetY } = getCoordinates(e);
    contextRef.current?.beginPath();
    contextRef.current?.moveTo(offsetX, offsetY);
    setIsDrawing(true);
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing) return;
    const { offsetX, offsetY } = getCoordinates(e);
    contextRef.current?.lineTo(offsetX, offsetY);
    contextRef.current?.stroke();
  };

  const stopDrawing = () => {
    contextRef.current?.closePath();
    setIsDrawing(false);
  };

  const getCoordinates = (e: React.MouseEvent | React.TouchEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return { offsetX: 0, offsetY: 0 };

    const rect = canvas.getBoundingClientRect();
    let clientX, clientY;

    if ('touches' in e) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }

    return {
      offsetX: clientX - rect.left,
      offsetY: clientY - rect.top
    };
  };

  const predict = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    setIsLoading(true);
    setError(null);

    try {
      // Create a temporary 28x28 canvas to downsample
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = MNIST_SIZE;
      tempCanvas.height = MNIST_SIZE;
      const tempCtx = tempCanvas.getContext('2d');
      
      if (!tempCtx) throw new Error('Could not create processing context');

      // Draw the large canvas onto the small one
      tempCtx.drawImage(canvas, 0, 0, MNIST_SIZE, MNIST_SIZE);

      // Get pixel data
      const imageData = tempCtx.getImageData(0, 0, MNIST_SIZE, MNIST_SIZE);
      const pixels = [];

      // MNIST is grayscale. We take the red channel (or alpha) since it's white on black.
      // Every 4th value is alpha (or we can use red/green/blue as they are same for white).
      for (let i = 0; i < imageData.data.length; i += 4) {
        // Normalize to 0-1
        pixels.push(imageData.data[i] / 255);
      }

      // Call FastAPI backend
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: pixels }),
      });

      if (!response.ok) {
        throw new Error('Prediction API failed. Ensure the FastAPI backend is running and model is trained.');
      }

      const result = await response.json();
      setPrediction({
        digit: result.prediction,
        confidence: result.confidence
      });
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : 'An error occurred during prediction');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-indigo-500/30">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <header className="mb-12 text-center">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center justify-center gap-3 mb-4 text-indigo-400"
          >
            <BrainCircuit size={40} className="animate-pulse" />
            <h1 className="text-4xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-cyan-400">
              Neural Digit
            </h1>
          </motion.div>
          <p className="text-slate-400 max-w-xl mx-auto">
            Draw a digit (0-9) in the box below and let the neural network classify it in real-time.
          </p>
        </header>

        <main className="grid md:grid-cols-2 gap-12 items-start">
          {/* Drawing Area */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex flex-col gap-6"
          >
            <div className="relative group">
              <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 to-cyan-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition duration-1000"></div>
              <canvas
                ref={canvasRef}
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={stopDrawing}
                onMouseLeave={stopDrawing}
                onTouchStart={startDrawing}
                onTouchMove={draw}
                onTouchEnd={stopDrawing}
                className="relative bg-black rounded-xl cursor-crosshair touch-none overflow-hidden block mx-auto border-2 border-slate-800"
                id="digit-canvas"
              />
              
              <div className="absolute top-4 right-4 text-xs font-mono text-slate-600 bg-black/50 px-2 py-1 rounded">
                28x28px Latent
              </div>
            </div>

            <div className="flex gap-4">
              <button
                onClick={clearCanvas}
                className="flex-1 flex items-center justify-center gap-2 py-3 px-6 rounded-xl bg-slate-900 border border-slate-800 hover:bg-slate-800 transition-colors font-medium text-slate-300"
                id="clear-btn"
              >
                <Eraser size={18} />
                Clear
              </button>
              <button
                onClick={predict}
                disabled={isLoading}
                className="flex-1 flex items-center justify-center gap-2 py-3 px-6 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-600/50 transition-colors font-bold text-white shadow-lg shadow-indigo-600/20 active:scale-95"
                id="predict-btn"
              >
                {isLoading ? (
                  <RefreshCw size={18} className="animate-spin" />
                ) : (
                  <Play size={18} />
                )}
                Classify
              </button>
            </div>
          </motion.div>

          {/* Results Area */}
          <div className="space-y-8">
            <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-8 backdrop-blur-sm">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 mb-6 flex items-center gap-2">
                <Activity size={16} />
                Network Output
              </h2>

              <AnimatePresence mode="wait">
                {prediction ? (
                  <motion.div
                    key="prediction"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    className="flex flex-col items-center"
                  >
                    <div className="text-8xl font-black text-indigo-400 mb-4 drop-shadow-[0_0_15px_rgba(129,140,248,0.4)]">
                      {prediction.digit}
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-2.5 mb-2 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${prediction.confidence * 100}%` }}
                        className="bg-indigo-500 h-full rounded-full"
                      />
                    </div>
                    <div className="text-sm font-mono text-slate-400">
                      Confidence: {(prediction.confidence * 100).toFixed(2)}%
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="placeholder"
                    className="flex flex-col items-center justify-center py-12 text-slate-600 text-center"
                  >
                    <Cpu size={64} className="mb-4 opacity-10" />
                    <p>Draw a number and click Classify</p>
                  </motion.div>
                )}
              </AnimatePresence>

              {error && (
                <div className="mt-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm">
                  {error}
                </div>
              )}
            </div>

            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-slate-400 px-2">How it works</h3>
              <div className="grid gap-3">
                {[
                  { title: 'Feature Extraction', desc: 'The 280px canvas is downsampled to MNIST-standard 28x28 grayscale.', icon: '1' },
                  { title: 'Vectorization', desc: 'The 784 pixels are flattened into a 1D feature vector for the model.', icon: '2' },
                  { title: 'Inference', desc: 'A Scikit-Learn MLP classifier processes the vector via its learned weights.', icon: '3' },
                ].map((step) => (
                  <div key={step.title} className="flex gap-4 p-4 rounded-xl hover:bg-slate-900/40 transition-colors group">
                    <div className="flex-none w-8 h-8 rounded-lg bg-slate-800 flex items-center justify-center text-xs font-bold text-slate-400 group-hover:bg-indigo-500/20 group-hover:text-indigo-400 transition-colors">
                      {step.icon}
                    </div>
                    <div>
                      <h4 className="text-sm font-medium text-slate-200">{step.title}</h4>
                      <p className="text-xs text-slate-500">{step.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </main>

        <footer className="mt-24 pt-8 border-t border-slate-900 text-center text-slate-600 text-sm">
          <p>© 2024 Neural Digit AI Studio • Scikit-Learn + FastAPI</p>
        </footer>
      </div>
    </div>
  );
}

