## SpecStory

## Extending Image Embedding Model Functionality (3/12/2025, 11:00:29 AM)

_**User**_

Right now, our project can only create embeddings of images using one embedding model, MobileNet. I want to extend the functionality of the project by allowing users to select from a variety of embedding models. Let's go step by step and update this app to allow this functionality

---

_**Assistant**_

# Adding Multiple Embedding Models to the Image Similarity App

I'll help you implement a dropdown menu that allows users to choose between different embedding models for image similarity comparison. This will involve creating a utility file for the models, updating the API routes, and modifying the UI.

## 1. Create a Utility File for Embedding Models

First, let's create a utility file to manage different embedding models:

```typescript:utils/embedding-models.ts
import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as blazeface from "@tensorflow-models/blazeface";

export type EmbeddingModel = {
  name: string;
  description: string;
  collectionName: string;
  vectorSize: number;
  load: () => Promise<any>;
  getEmbedding: (
    model: any,
    imageElement: HTMLImageElement
  ) => Promise<Float32Array>;
};

// Cache for loaded models
const modelCache: Record<string, any> = {};

export const EMBEDDING_MODELS: EmbeddingModel[] = [
  {
    name: "MobileNet",
    description: "Lightweight model optimized for mobile devices",
    collectionName: "mobilenet_embeddings",
    vectorSize: 1280,
    load: async () => {
      if (!modelCache.mobilenet) {
        modelCache.mobilenet = await mobilenet.load();
      }
      return modelCache.mobilenet;
    },
    getEmbedding: async (model, imageElement) => {
      const tfImage = tf.browser.fromPixels(imageElement);
      const resized = tf.image.resizeBilinear(tfImage, [224, 224]);
      const expanded = resized.expandDims(0);
      const normalized = expanded.div(255);

      const embedding = await model.infer(normalized, {
        embedding: true,
      });

      const embeddingData = await embedding.data();

      // Cleanup tensors
      tfImage.dispose();
      resized.dispose();
      expanded.dispose();
      normalized.dispose();
      embedding.dispose();

      return new Float32Array(embeddingData);
    },
  },
  {
    name: "BlazeFace",
    description: "Face detection model with feature extraction",
    collectionName: "blazeface_embeddings",
    vectorSize: 256,
    load: async () => {
      if (!modelCache.blazeface) {
        modelCache.blazeface = await blazeface.load();
      }
      return modelCache.blazeface;
    },
    getEmbedding: async (model, imageElement) => {
      const tfImage = tf.browser.fromPixels(imageElement);
      const resized = tf.image.resizeBilinear(tfImage, [224, 224]);
      const expanded = resized.expandDims(0);
      const normalized = expanded.div(255);

      // Get face predictions
      const predictions = await model.estimateFaces(imageElement, false);
      
      // If no faces detected, use a generic embedding from the image
      if (predictions.length === 0) {
        // Create a simple embedding from the image data
        const flattened = normalized.reshape([224 * 224 * 3]);
        const pooled = tf.pool(flattened.reshape([1, 224, 224, 3]), [2, 2], 'avg', 'valid');
        const embeddingTensor = pooled.reshape([256]);
        const embeddingData = await embeddingTensor.data();
        
        // Cleanup
        tfImage.dispose();
        resized.dispose();
        expanded.dispose();
        normalized.dispose();
        flattened.dispose();
        pooled.dispose();
        embeddingTensor.dispose();
        
        return new Float32Array(embeddingData);
      }
      
      // Use the first face's topLeft and bottomRight to extract face region
      const face = predictions[0];
      const [x1, y1] = face.topLeft;
      const [x2, y2] = face.bottomRight;
      
      // Extract face region and create embedding
      const faceBox = tf.image.cropAndResize(
        normalized, 
        [[y1/224, x1/224, y2/224, x2/224]], 
        [0], 
        [16, 16]
      );
      
      const flattenedFace = faceBox.reshape([16 * 16 * 3]);
      const embeddingTensor = tf.slice(flattenedFace, 0, 256);
      const embeddingData = await embeddingTensor.data();
      
      // Cleanup
      tfImage.dispose();
      resized.dispose();
      expanded.dispose();
      normalized.dispose();
      faceBox.dispose();
      flattenedFace.dispose();
      embeddingTensor.dispose();
      
      return new Float32Array(embeddingData);
    },
  },
];
```

## 2. Update the API Route for Qdrant Operations

Now, let's update the API route to handle different collections based on the selected model:

```typescript:app/api/qdrant/route.ts
import { NextRequest, NextResponse } from "next/server";
import { QdrantClient } from "@qdrant/js-client-rest";
import crypto from "crypto";
import { EMBEDDING_MODELS } from "@/utils/embedding-models";

const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
});

function generatePointId(vector: number[]): string {
  return crypto
    .createHash('sha256')
    .update(vector.join(','))
    .digest('hex');
}

async function ensureCollection(collectionName: string, vectorSize: number) {
  try {
    const collections = await qdrant.getCollections();
    const exists = collections.collections.some(
      (collection) => collection.name === collectionName
    );

    if (!exists) {
      await qdrant.createCollection(collectionName, {
        vectors: {
          size: vectorSize,
          distance: "Cosine",
        },
      });
    }
  } catch (error) {
    console.error("Error ensuring collection exists:", error);
    throw error;
  }
}

export async function POST(request: NextRequest) {
  try {
    const { action, modelName, data } = await request.json();
    
    const model = EMBEDDING_MODELS.find(m => m.name === modelName);
    if (!model) {
      return NextResponse.json({ error: "Invalid model" }, { status: 400 });
    }

    await ensureCollection(model.collectionName, model.vectorSize);

    switch (action) {
      case "upsert":
        const { vector, metadata } = data;
        const pointId = generatePointId(vector);
        await qdrant.upsert(model.collectionName, {
          points: [
            {
              id: pointId,
              vector,
              payload: {
                ...metadata,
                id: pointId,
                modelName,
              },
            },
          ],
        });
        return NextResponse.json({ success: true });

      case "search":
        const { queryVector, limit = 5 } = data;
        const results = await qdrant.search(model.collectionName, {
          vector: queryVector,
          limit,
        });
        return NextResponse.json({ results });

      default:
        return NextResponse.json({ error: "Invalid action" }, { status: 400 });
    }
  } catch (error) {
    console.error("Qdrant operation error:", error);
    return NextResponse.json(
      { error: "Failed to perform Qdrant operation" },
      { status: 500 }
    );
  }
}
```

## 3. Update the Main Page Component

Now, let's update the page component to include model selection:

```typescript:app/page.tsx
"use client";

import { useState } from "react";
import Image from "next/image";
import { EMBEDDING_MODELS, type EmbeddingModel } from "@/utils/embedding-models";

type UploadStatus = {
  filename: string;
  status: "pending" | "processing" | "complete" | "error";
  error?: string;
};

export default function Home() {
  const [selectedModel, setSelectedModel] = useState<EmbeddingModel>(EMBEDDING_MODELS[0]);
  const [collectionImages, setCollectionImages] = useState<File[]>([]);
  const [uploadStatuses, setUploadStatuses] = useState<UploadStatus[]>([]);
  const [targetImage, setTargetImage] = useState<File | null>(null);
  const [targetPreview, setTargetPreview] = useState<string | null>(null);
  const [similarityScore, setSimilarityScore] = useState<number | null>(null);
  const [isComparing, setIsComparing] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleCollectionUpload = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const files = Array.from(e.target.files || []);
    setCollectionImages(files);
    setUploadStatuses(
      files.map((file) => ({
        filename: file.name,
        status: "pending",
      }))
    );
  };

  const processCollection = async () => {
    setIsProcessing(true);
    
    let model;
    try {
      model = await selectedModel.load();
    } catch (error) {
      console.error("Error loading model:", error);
      setIsProcessing(false);
      return;
    }
    
    try {
      for (let i = 0; i < collectionImages.length; i++) {
        const file = collectionImages[i];
        setUploadStatuses((prev) =>
          prev.map((status, idx) =>
            idx === i ? { ...status, status: "processing" } : status
          )
        );

        try {
          // Create an image element and load the file
          const img = document.createElement("img");
          img.src = URL.createObjectURL(file);
          await new Promise((resolve) => {
            img.onload = resolve;
          });

          // Get embedding using the selected model
          const embedding = await selectedModel.getEmbedding(model, img);

          // Store in Qdrant
          const response = await fetch("/api/qdrant", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              action: "upsert",
              modelName: selectedModel.name,
              data: {
                vector: Array.from(embedding),
                metadata: {
                  filename: file.name,
                  uploadedAt: new Date().toISOString(),
                },
              },
            }),
          });

          if (!response.ok) throw new Error("Failed to store embedding");

          setUploadStatuses((prev) =>
            prev.map((status, idx) =>
              idx === i ? { ...status, status: "complete" } : status
            )
          );

          // Cleanup
          URL.revokeObjectURL(img.src);
        } catch (error: unknown) {
          setUploadStatuses((prev) =>
            prev.map((status, idx) =>
              idx === i
                ? {
                    ...status,
                    status: "error",
                    error:
                      error instanceof Error ? error.message : "Unknown error",
                  }
                : status
            )
          );
        }
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleTargetImageUpload = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0];
    if (file) {
      setTargetImage(file);
      setTargetPreview(URL.createObjectURL(file));
      setSimilarityScore(null);
    }
  };

  const handleCompare = async () => {
    if (!targetImage) return;

    setIsComparing(true);
    try {
      // Load the model
      const model = await selectedModel.load();
      
      // Create an image element and load the target file
      const img = document.createElement('img');
      img.src = URL.createObjectURL(targetImage);
      await new Promise((resolve) => {
        img.onload = resolve;
      });

      // Get embedding for target image
      const embedding = await selectedModel.getEmbedding(model, img);

      // Search for similar images in Qdrant
      const response = await fetch("/api/qdrant", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          action: "search",
          modelName: selectedModel.name,
          data: {
            queryVector: Array.from(embedding),
            limit: 1,
          },
        }),
      });

      const data = await response.json();
      const similarity = data.results?.[0]?.score || 0;
      setSimilarityScore(similarity);
      
      // Cleanup
      URL.revokeObjectURL(img.src);
    } catch (error) {
      console.error("Error comparing images:", error);
    } finally {
      setIsComparing(false);
    }
  };

  return (
    <div className="min-h-screen p-8">
      <main className="max-w-4xl mx-auto space-y-12">
        {/* Model Selection */}
        <section className="space-y-4">
          <h2 className="text-2xl font-bold">Select Embedding Model</h2>
          <select
            value={selectedModel.name}
            onChange={(e) => {
              const model = EMBEDDING_MODELS.find(m => m.name === e.target.value);
              if (model) setSelectedModel(model);
            }}
            className="w-full p-2 border rounded-md"
          >
            {EMBEDDING_MODELS.map((model) => (
              <option key={model.name} value={model.name}>
                {model.name} - {model.description}
              </option>
            ))}
          </select>
        </section>

        {/* Collection Building Section */}
        <section className="space-y-6">
          <h2 className="text-2xl font-bold">Add Image(s) to Collection</h2>
          <input
            type="file"
            multiple
            accept="image/*"
            onChange={handleCollectionUpload}
            className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-violet-50 file:text-violet-700
              hover:file:bg-violet-100"
          />

          {uploadStatuses.length > 0 && (
            <div className="space-y-4">
              <button
                onClick={processCollection}
                disabled={isProcessing}
                className="px-4 py-2 bg-violet-500 text-white rounded-md
                  hover:bg-violet-600 cursor-grab active:cursor-grabbing
                  disabled:bg-violet-300 disabled:cursor-not-allowed"
              >
                {isProcessing ? "Processing Images..." : "Process Images"}
              </button>

              <div className="space-y-2">
                {uploadStatuses.map((status, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <span>{status.filename}</span>
                    <span
                      className={`text-sm ${
                        status.status === "complete"
                          ? "text-green-500"
                          : status.status === "error"
                          ? "text-red-500"
                          : status.status === "processing"
                          ? "text-blue-500"
                          : "text-gray-500"
                      }`}
                    >
                      {status.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        {/* Image Comparison Section */}
        <section className="space-y-6">
          <h2 className="text-2xl font-bold">Compare Image</h2>
          <input
            type="file"
            accept="image/*"
            onChange={handleTargetImageUpload}
            className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-50 file:text-blue-700
              hover:file:bg-blue-100"
          />

          {targetPreview && (
            <div className="space-y-4">
              <Image
                src={targetPreview}
                alt="Preview"
                width={300}
                height={300}
                className="object-contain"
              />

              <button
                onClick={handleCompare}
                disabled={isComparing}
                className="px-4 py-2 bg-blue-500 text-white rounded-md
                  disabled:bg-gray-300 disabled:cursor-not-allowed"
              >
                {isComparing ? "Comparing..." : "Get Similarity Score"}
              </button>
            </div>
          )}

          {similarityScore !== null && (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
              <p className="text-lg text-black font-semibold">
                Similarity Score: {(similarityScore * 100).toFixed(2)}%
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
```

## 4. Install Required Dependencies

You'll need to install the BlazeFace package:

```bash
npm install @tensorflow-models/blazeface
```

## Key Features of This Implementation

1. **Model Selection**: Users can choose between different embedding models via a dropdown menu.

2. **Separate Collections**: Each model has its own collection in Qdrant, ensuring embeddings from different models don't mix.

3. **Client-Side Processing**: All embedding generation happens in the browser, reducing server load.

4. **Model Caching**: Models are cached after first load to improve performance.

5. **Flexible Architecture**: New models can be easily added by extending the `EMBEDDING_MODELS` array.

The implementation provides two distinct models:

1. **MobileNet**: A general-purpose image classification model that works well for most images.

2. **BlazeFace**: A face detection model that can be useful for comparing faces in images.

This approach allows users to experiment with different embedding techniques and compare their effectiveness for different types of images.

---

_**User**_

I want to reuse my existing API route handlers to add image embeddings to qdrant. we just need to create separate collections for each model

---

_**Assistant**_

You're right - we can reuse the existing API routes and just modify them to handle different collections based on the selected model. Let's update the necessary files:

```typescript:utils/embedding-node.ts
import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as blazeface from "@tensorflow-models/blazeface";
import sharp from "sharp";

// Cache for loaded models
const modelCache: Record<string, any> = {};

export type EmbeddingModelType = "MobileNet" | "BlazeFace";

export async function loadModel(modelType: EmbeddingModelType) {
  const cacheKey = modelType.toLowerCase();
  
  if (!modelCache[cacheKey]) {
    // Use CPU backend
    await tf.setBackend("cpu");
    
    if (modelType === "MobileNet") {
      modelCache[cacheKey] = await mobilenet.load({ version: 2, alpha: 1.0 });
    } else if (modelType === "BlazeFace") {
      modelCache[cacheKey] = await blazeface.load();
    }
  }
  
  return modelCache[cacheKey];
}

export async function generateEmbedding(
  base64Image: string,

---



---

