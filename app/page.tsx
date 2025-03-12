"use client";

import { useState } from "react";
import Image from "next/image";

type UploadStatus = {
  filename: string;
  status: "pending" | "processing" | "complete" | "error";
  error?: string;
};

export default function Home() {
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
    try {
      for (let i = 0; i < collectionImages.length; i++) {
        const file = collectionImages[i];
        setUploadStatuses((prev) =>
          prev.map((status, idx) =>
            idx === i ? { ...status, status: "processing" } : status
          )
        );

        try {
          const formData = new FormData();
          formData.append("image", file);

          const response = await fetch("/api/add-to-collection", {
            method: "POST",
            body: formData,
          });

          if (!response.ok) throw new Error("Failed to process image");

          setUploadStatuses((prev) =>
            prev.map((status, idx) =>
              idx === i ? { ...status, status: "complete" } : status
            )
          );
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
      const formData = new FormData();
      formData.append("image", targetImage);

      const response = await fetch("/api/compare-image", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setSimilarityScore(data.similarity);
    } catch (error) {
      console.error("Error comparing images:", error);
    } finally {
      setIsComparing(false);
    }
  };

  return (
    <div className="min-h-screen p-8">
      <main className="max-w-4xl mx-auto space-y-12">
        {/* Collection Building Section */}
        <section className="space-y-6">
          <h2 className="text-2xl font-bold">Add Image(s) toCollection</h2>
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
