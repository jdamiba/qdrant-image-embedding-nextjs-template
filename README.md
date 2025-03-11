# Image Similarity Search with TensorFlow.js and Qdrant

This application demonstrates how to implement image similarity search using MobileNet for generating vector embeddings and Qdrant for similarity search.

## Technical Overview

### Vector Embeddings with TensorFlow.js

- Uses MobileNet V2 model through TensorFlow.js to generate vector embeddings (1028-dimensional vectors) from images
- Runs on CPU backend for compatibility with serverless environments
- The embedding process:
  1. Resizes images to 224x224 pixels (MobileNet's expected input size)
  2. Processes images using Sharp for consistent formatting
  3. Passes the image through MobileNet to generate embeddings
  4. Uses the model's feature vectors as embeddings for similarity comparison

### Similarity Search with Qdrant

- Stores image embeddings in a Qdrant collection with Cosine similarity metric
- Each vector in the collection includes:
  - 1028-dimensional embedding vector
  - Metadata (filename)
- Performs similarity search using Qdrant's vector search capabilities
- Returns similarity scores between 0 and 1 (higher means more similar)

## Getting Started

1. Set up environment variables:

```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

2. Install dependencies:

```bash
npm install
```

3. Run the development server:

```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) with your browser.

## How It Works

1. **Adding Images to Collection**

   - Upload one or more images
   - TensorFlow.js generates embeddings using MobileNet
   - Embeddings are stored in Qdrant collection

2. **Finding Similar Images**
   - Upload a target image
   - Generate its embedding
   - Query Qdrant to find similarity scores
   - Display the similarity percentage

## Technical Details

### Image Processing Pipeline

- Uses Sharp for consistent image preprocessing
- Converts images to raw pixel data for TensorFlow.js
- Maintains aspect ratios while resizing to required dimensions

### Vector Search

- Collection uses Cosine similarity for comparing vectors
- Each vector has 1028 dimensions (MobileNet V2's feature vector size)
- Similarity scores are normalized between 0 and 1

## Learn More

- [TensorFlow.js Documentation](https://www.tensorflow.org/js)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [MobileNet Model](https://github.com/tensorflow/tfjs-models/tree/master/mobilenet)
- [Sharp Image Processing](https://sharp.pixelplumbing.com/)
