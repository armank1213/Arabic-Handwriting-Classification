'use client'

import React, { useState, useRef, useEffect } from 'react';
import { Button } from 'app/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from 'app/components/ui/card';

interface Prediction {
  arabic_char: string;
  english_name: string;
  confidence: string;
}

const LetterClassificationApp = () => {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const displayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    const displayCanvas = displayCanvasRef.current;
    if (canvas && displayCanvas) {
      const context = canvas.getContext('2d');
      const displayContext = displayCanvas.getContext('2d');
      if (context && displayContext) {
        context.lineWidth = 2;
        context.lineCap = 'round';
        context.strokeStyle = 'white';
        context.fillStyle = 'black';
        context.fillRect(0, 0, canvas.width, canvas.height);

        displayContext.lineWidth = 8;
        displayContext.lineCap = 'round';
        displayContext.strokeStyle = 'white';
        displayContext.fillStyle = 'black';
        displayContext.fillRect(0, 0, displayCanvas.width, displayCanvas.height);
      }
    }
  }, []);

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true);
    draw(e);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    const canvas = canvasRef.current;
    const displayCanvas = displayCanvasRef.current;
    if (canvas && displayCanvas) {
      const ctx = canvas.getContext('2d');
      const displayCtx = displayCanvas.getContext('2d');
      if (ctx && displayCtx) {
        ctx.beginPath();
        displayCtx.beginPath();
      }
    }
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const displayCanvas = displayCanvasRef.current;
    if (canvas && displayCanvas) {
      const rect = displayCanvas.getBoundingClientRect();
      const ctx = canvas.getContext('2d');
      const displayCtx = displayCanvas.getContext('2d');
      if (ctx && displayCtx) {
        const x = (e.clientX - rect.left) * (canvas.width / displayCanvas.width);
        const y = (e.clientY - rect.top) * (canvas.height / displayCanvas.height);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);

        const displayX = e.clientX - rect.left;
        const displayY = e.clientY - rect.top;
        displayCtx.lineTo(displayX, displayY);
        displayCtx.stroke();
        displayCtx.beginPath();
        displayCtx.moveTo(displayX, displayY);
      }
    }
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const displayCanvas = displayCanvasRef.current;
    if (canvas && displayCanvas) {
      const ctx = canvas.getContext('2d');
      const displayCtx = displayCanvas.getContext('2d');
      if (ctx && displayCtx) {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        displayCtx.fillStyle = 'black';
        displayCtx.fillRect(0, 0, displayCanvas.width, displayCanvas.height);
        setPredictions([]);
      }
    }
  };

  const classifyLetter = async () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const imageData = canvas.toDataURL('image/png');
      
      try {
        const response = await fetch('http://localhost:8000/classify', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ image: imageData }),
        });
        
        if (!response.ok) {
          throw new Error('Classification failed');
        }
        
        const result = await response.json();
        setPredictions(result);
      } catch (error) {
        console.error('Error:', error);
        setPredictions([]);
      }
    }
  };

  useEffect(() => {
    clearCanvas();
  }, []);

  return (
    <div className="flex items-center justify-center min-h-screen bg-neutral-900 text-white">
      <div className="flex space-x-8">
        <Card className="w-96 bg-neutral-800 border-neutral-700">
          <CardHeader>
            <CardTitle className="text-white">Arabic Letter Classification</CardTitle>
          </CardHeader>
          <CardContent>
            <canvas
              ref={displayCanvasRef}
              width={280}
              height={280}
              className="border border-neutral-600 rounded"
              onMouseDown={startDrawing}
              onMouseUp={stopDrawing}
              onMouseOut={stopDrawing}
              onMouseMove={draw}
            />
            <canvas
              ref={canvasRef}
              width={32}
              height={32}
              className="hidden"
            />
            <div className="flex justify-between mt-4">
              <Button className='bg-neutral-200 hover:bg-neutral-100 text-black' onClick={clearCanvas}>Clear</Button>
              <Button className='bg-blue-500 hover:bg-blue-400' onClick={classifyLetter}>Classify</Button>
            </div>
          </CardContent>
        </Card>
        
        <div className="w-64">
          <h3 className="text-xl font-bold mb-4">Top 3 Predictions:</h3>
          {predictions.length > 0 ? (
            predictions.map((pred, index) => (
              <div key={index} className="mb-4 p-4 bg-neutral-800 rounded-lg">
                <p className="text-3xl mb-1">{pred.arabic_char}</p>
                <p className="text-lg mb-1">{pred.english_name}</p>
                <p className="text-sm text-neutral-400">Confidence: {pred.confidence}</p>
              </div>
            ))
          ) : (
            <p className="text-neutral-400">Draw a letter and click 'Classify' to see predictions</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default LetterClassificationApp;