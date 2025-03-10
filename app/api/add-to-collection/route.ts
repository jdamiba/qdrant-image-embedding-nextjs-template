import { NextRequest, NextResponse } from "next/server";
import { QdrantClient } from "@qdrant/js-client-rest";
import { generateEmbedding } from "@/utils/embedding-node";

const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
});
const COLLECTION_NAME = "image_vector_embeddings_20250310";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("image") as File;

    if (!file) {
      return NextResponse.json(
        { error: "No image file provided" },
        { status: 400 }
      );
    }

    // Convert file to base64
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const base64Image = buffer.toString("base64");

    // Generate embedding
    const embedding = await generateEmbedding(base64Image);

    // Upload to Qdrant
    await qdrant.upsert(COLLECTION_NAME, {
      points: [
        {
          id: Date.now(), // Using timestamp as ID - you might want a better ID strategy
          vector: embedding,
          payload: {
            filename: file.name,
          },
        },
      ],
    });

    return NextResponse.json({ success: true });
  } catch (error: any) {
    console.error("Error processing image:", error);
    return NextResponse.json(
      { error: error.message || "Error processing image" },
      { status: 500 }
    );
  }
}
