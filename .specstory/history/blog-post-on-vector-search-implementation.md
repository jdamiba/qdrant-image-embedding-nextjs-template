## SpecStory

## Blog Post on Vector Search Implementation (3/9/2025, 11:48:26 PM)

_**User**_

given the code in this project, write a blog post with the title "Implementing Vector Search in Next.js Using Qdrant and Google Tensorflow". 

---

_**Assistant**_

# Implementing Vector Search in Next.js Using Qdrant and Google TensorFlow

Vector search has revolutionized how we find similar content across various domains, from images to text. In this post, I'll walk through implementing an image similarity search application using Next.js, TensorFlow.js, and Qdrant - a powerful vector database.

## What We're Building

We'll create an application that:
1. Takes images uploaded by users
2. Converts them into vector embeddings using TensorFlow.js and MobileNet
3. Stores these embeddings in Qdrant
4. Allows users to find similar images by uploading a target image

## The Technology Stack

- **Next.js**: Our React framework for building the application
- **TensorFlow.js**: For generating vector embeddings from images
- **MobileNet**: A pre-trained neural network for feature extraction
- **Qdrant**: A vector database optimized for similarity search
- **Sharp**: For image preprocessing

## Understanding Vector Embeddings

Vector embeddings are numerical representations of data that capture semantic meaning. For images, these embeddings represent visual features like shapes, textures, and objects.

When we run an image through MobileNet, it generates a 1280-dimensional vector that represents the image's features. Similar images will have vectors that are close to each other in this high-dimensional space.

## Setting Up the Project

First, we need to set up our environment with the necessary dependencies:

```bash
npm install @tensorflow/tfjs @qdrant/js-client-rest sharp
```

And configure our environment variables:

```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

## Generating Image Embeddings

The core of our application is the embedding generation process. We use TensorFlow.js with the MobileNet model to extract feature vectors from images:

```typescript
// utils/embedding-node.ts
import * as tf from '@tensorflow/tfjs-node';
import sharp from 'sharp';

let model: tf.GraphModel;

async function loadModel() {
  if (!model) {
    model = await tf.loadGraphModel(
      'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/feature_vector/2/default/1',
      { fromTFHub: true }
    );
  }
  return model;
}

export async function generateEmbedding(base64Image: string): Promise<number[]> {
  // Load the model
  const model = await loadModel();
  
  // Decode and preprocess the image
  const imageBuffer = Buffer.from(base64Image, 'base64');
  const { data, info } = await sharp(imageBuffer)
    .resize(224, 224, { fit: 'contain', background: { r: 255, g: 255, b: 255 } })
    .raw()
    .toBuffer({ resolveWithObject: true });
  
  // Convert to tensor
  const tensor = tf.tensor3d(
    new Uint8Array(data),
    [224, 224, info.channels],
    'int32'
  );
  
  // Normalize and reshape
  const normalized = tf.div(tensor, 255);
  const batched = tf.expandDims(normalized, 0);
  
  // Generate embedding
  const embedding = model.predict(batched) as tf.Tensor;
  
  // Convert to array and return
  const embeddingArray = await embedding.data();
  return Array.from(embeddingArray);
}
```

## Setting Up Qdrant for Vector Storage

Next, we need to set up our Qdrant collection to store these embeddings:

```typescript
// app/api/add-to-collection/route.ts
import { QdrantClient } from "@qdrant/js-client-rest";

const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
});

const COLLECTION_NAME = "image_vector_embeddings_20250310";
const VECTOR_SIZE = 1280; // MobileNet embedding size

async function ensureCollection() {
  try {
    // Check if collection exists
    const collections = await qdrant.getCollections();
    const exists = collections.collections.some(
      (collection) => collection.name === COLLECTION_NAME
    );

    if (!exists) {
      // Create collection if it doesn't exist
      await qdrant.createCollection(COLLECTION_NAME, {
        vectors: {
          size: VECTOR_SIZE,
          distance: "Cosine",
        },
      });
    }
  } catch (error) {
    console.error("Error ensuring collection exists:", error);
    throw error;
  }
}
```

## Adding Images to the Collection

When a user uploads an image, we process it and add it to our Qdrant collection:

```typescript
// app/api/add-to-collection/route.ts
export async function POST(request: NextRequest) {
  try {
    // Ensure collection exists before proceeding
    await ensureCollection();

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
          id: Date.now(),
          vector: embedding,
          payload: {
            filename: file.name,
          },
        },
      ],
    });

    return NextResponse.json({ success: true });
  } catch (error: unknown) {
    console.error("Error processing image:", error);
    // Error handling...
  }
}
```

## Searching for Similar Images

To find similar images, we generate an embedding for the query image and search for the closest vectors in our Qdrant collection:

```typescript
// app/api/search/route.ts
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
    
    // Generate embedding for the query image
    const embedding = await generateEmbedding(base64Image);
    
    // Search for similar images in Qdrant
    const searchResults = await qdrant.search(COLLECTION_NAME, {
      vector: embedding,
      limit: 5,  // Return top 5 matches
      with_payload: true,
    });
    
    return NextResponse.json({ results: searchResults });
  } catch (error: unknown) {
    // Error handling...
  }
}
```

## Building the Frontend

For the frontend, we'll create a simple interface with two main components:

1. An upload form for adding images to our collection
2. A search form for finding similar images

```tsx
// app/page.tsx
'use client';

import { useState } from 'react';
import Image from 'next/image';

export default function Home() {
  const [uploadStatus, setUploadStatus] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);

  async function handleUpload(e) {
    e.preventDefault();
    const formData = new FormData(e.target);
    
    try {
      setUploadStatus('Uploading...');
      const response = await fetch('/api/add-to-collection', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        setUploadStatus('Upload successful!');
      } else {
        setUploadStatus('Upload failed.');
      }
    } catch (error) {
      setUploadStatus('Error uploading image.');
    }
  }

  async function handleSearch(e) {
    e.preventDefault();
    const formData = new FormData(e.target);
    
    try {
      setIsSearching(true);
      const response = await fetch('/api/search', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const data = await response.json();
        setSearchResults(data.results);
      }
    } catch (error) {
      console.error('Error searching:', error);
    } finally {
      setIsSearching(false);
    }
  }

  return (
    <main className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">Image Similarity Search</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="border p-4 rounded">
          <h2 className="text-xl mb-4">Add Images to Collection</h2>
          <form onSubmit={handleUpload}>
            <input type="file" name="image" accept="image/*" required />
            <button type="submit" className="mt-2 px-4 py-2 bg-blue-500 text-white rounded">
              Upload
            </button>
          </form>
          {uploadStatus && <p className="mt-2">{uploadStatus}</p>}
        </div>
        
        <div className="border p-4 rounded">
          <h2 className="text-xl mb-4">Find Similar Images</h2>
          <form onSubmit={handleSearch}>
            <input type="file" name="image" accept="image/*" required />
            <button 
              type="submit" 
              className="mt-2 px-4 py-2 bg-green-500 text-white rounded"
              disabled={isSearching}
            >
              {isSearching ? 'Searching...' : 'Search'}
            </button>
          </form>
          
          {searchResults.length > 0 && (
            <div className="mt-4">
              <h3 className="text-lg font-semibold">Results:</h3>
              <ul className="mt-2">
                {searchResults.map((result, index) => (
                  <li key={index} className="mb-2">
                    <p>Filename: {result.payload.filename}</p>
                    <p>Similarity: {(result.score * 100).toFixed(2)}%</p>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
```

## Performance Considerations

When implementing vector search, keep these performance considerations in mind:

1. **Model Loading**: TensorFlow.js models can be large. Load them once and reuse the instance.
2. **Image Preprocessing**: Resize and normalize images consistently for accurate embeddings.
3. **Vector Dimensions**: Higher dimensions capture more features but require more storage and computation.
4. **Similarity Metrics**: We use Cosine similarity, but Euclidean or Dot Product might be better for certain applications.

## Scaling Considerations

As your application grows:

1. **Batch Processing**: Process multiple images in batches for efficiency.
2. **Caching**: Cache embeddings for frequently accessed images.
3. **Filtering**: Implement pre-filtering based on metadata before vector search.
4. **Hybrid Search**: Combine vector search with traditional filtering for better results.

## Conclusion

Vector search with TensorFlow.js and Qdrant opens up powerful possibilities for content discovery and recommendation systems. This implementation demonstrates how to build a complete image similarity search system using modern web technologies.

The beauty of this approach is its flexibility - you can easily swap MobileNet for other models to extract different features or adapt the system for other content types like text or audio.

By leveraging the power of vector embeddings, we can create intuitive search experiences that understand the semantic meaning of content rather than relying on exact matches or keywords.

Happy coding!

---

_**User**_

let's enhance the text prior to each codeblock with more explanation of how the code works. assume readers are novices in both nextjs and vector search

---

_**Assistant**_

# Implementing Vector Search in Next.js Using Qdrant and Google TensorFlow

Vector search has revolutionized how we find similar content across various domains, from images to text. In this post, I'll walk through implementing an image similarity search application using Next.js, TensorFlow.js, and Qdrant - a powerful vector database.

## What We're Building

We'll create an application that:
1. Takes images uploaded by users
2. Converts them into vector embeddings using TensorFlow.js and MobileNet
3. Stores these embeddings in Qdrant
4. Allows users to find similar images by uploading a target image

## The Technology Stack

- **Next.js**: Our React framework for building the application
- **TensorFlow.js**: For generating vector embeddings from images
- **MobileNet**: A pre-trained neural network for feature extraction
- **Qdrant**: A vector database optimized for similarity search
- **Sharp**: For image preprocessing

## Understanding Vector Embeddings

Vector embeddings are numerical representations of data that capture semantic meaning. For images, these embeddings represent visual features like shapes, textures, and objects.

When we run an image through MobileNet, it generates a 1280-dimensional vector that represents the image's features. Similar images will have vectors that are close to each other in this high-dimensional space.

Think of it like this: each image gets converted into a list of 1,280 numbers. Images of cats will have similar number patterns, while images of cars will have a different pattern. When we search, we're looking for images whose number patterns are closest to our query image.

## Setting Up the Project

First, we need to set up our environment with the necessary dependencies:

```bash
npm install @tensorflow/tfjs @qdrant/js-client-rest sharp
```

This command installs three key packages:
- `@tensorflow/tfjs`: The JavaScript version of TensorFlow that allows us to run machine learning models in our application
- `@qdrant/js-client-rest`: A client library to interact with the Qdrant vector database
- `sharp`: A high-performance image processing library for Node.js

And configure our environment variables:

```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

These environment variables are crucial for connecting to your Qdrant instance. You'll need to create an account with Qdrant Cloud or set up your own instance, then copy the URL and API key to these variables.

## Generating Image Embeddings

The core of our application is the embedding generation process. We use TensorFlow.js with the MobileNet model to extract feature vectors from images:

```typescript
// utils/embedding-node.ts
import * as tf from '@tensorflow/tfjs-node';
import sharp from 'sharp';

let model: tf.GraphModel;

async function loadModel() {
  if (!model) {
    model = await tf.loadGraphModel(
      'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/feature_vector/2/default/1',
      { fromTFHub: true }
    );
  }
  return model;
}

export async function generateEmbedding(base64Image: string): Promise<number[]> {
  // Load the model
  const model = await loadModel();
  
  // Decode and preprocess the image
  const imageBuffer = Buffer.from(base64Image, 'base64');
  const { data, info } = await sharp(imageBuffer)
    .resize(224, 224, { fit: 'contain', background: { r: 255, g: 255, b: 255 } })
    .raw()
    .toBuffer({ resolveWithObject: true });
  
  // Convert to tensor
  const tensor = tf.tensor3d(
    new Uint8Array(data),
    [224, 224, info.channels],
    'int32'
  );
  
  // Normalize and reshape
  const normalized = tf.div(tensor, 255);
  const batched = tf.expandDims(normalized, 0);
  
  // Generate embedding
  const embedding = model.predict(batched) as tf.Tensor;
  
  // Convert to array and return
  const embeddingArray = await embedding.data();
  return Array.from(embeddingArray);
}
```

Let's break down what this code does:

1. **Model Loading**: We load the MobileNet model from TensorFlow Hub. We use a singleton pattern (checking if `model` exists before loading) to avoid loading the model multiple times, which would be inefficient.

2. **Image Preprocessing**: 
   - We convert the base64 image string back to a buffer
   - Using Sharp, we resize the image to 224x224 pixels (the size MobileNet expects)
   - We convert the image to a raw pixel buffer

3. **Tensor Creation**: 
   - We create a TensorFlow tensor (a multi-dimensional array) from the image data
   - We normalize the pixel values by dividing by 255 (converting from 0-255 to 0-1 range)
   - We add a batch dimension because the model expects batches of images

4. **Embedding Generation**: 
   - We pass the processed image through the model with `model.predict()`
   - This returns a tensor containing the 1280-dimensional embedding
   - We convert this tensor to a JavaScript array and return it

This function is the heart of our application - it transforms visual data into numerical vectors that can be compared mathematically.

## Setting Up Qdrant for Vector Storage

Next, we need to set up our Qdrant collection to store these embeddings:

```typescript
// app/api/add-to-collection/route.ts
import { QdrantClient } from "@qdrant/js-client-rest";

const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
});

const COLLECTION_NAME = "image_vector_embeddings_20250310";
const VECTOR_SIZE = 1280; // MobileNet embedding size

async function ensureCollection() {
  try {
    // Check if collection exists
    const collections = await qdrant.getCollections();
    const exists = collections.collections.some(
      (collection) => collection.name === COLLECTION_NAME
    );

    if (!exists) {
      // Create collection if it doesn't exist
      await qdrant.createCollection(COLLECTION_NAME, {
        vectors: {
          size: VECTOR_SIZE,
          distance: "Cosine",
        },
      });
    }
  } catch (error) {
    console.error("Error ensuring collection exists:", error);
    throw error;
  }
}
```

This code sets up our connection to Qdrant and ensures we have a place to store our vectors:

1. **Client Initialization**: We create a new Qdrant client using our environment variables for authentication.

2. **Collection Configuration**: 
   - We define a collection name (like a table in a traditional database)
   - We specify the vector size (1280, matching MobileNet's output)
   - We choose "Cosine" as our distance metric, which measures the angle between vectors and is ideal for comparing image embeddings

3. **Collection Creation**: 
   - The `ensureCollection` function checks if our collection already exists
   - If it doesn't exist, it creates a new collection with our specified parameters
   - This is similar to creating a table in a database if it doesn't already exist

Think of a Qdrant collection as a specialized database table designed to store and efficiently search through vectors. The "Cosine" distance metric helps us find vectors pointing in similar directions, which correlates with image similarity.

## Adding Images to the Collection

When a user uploads an image, we process it and add it to our Qdrant collection:

```typescript
// app/api/add-to-collection/route.ts
export async function POST(request: NextRequest) {
  try {
    // Ensure collection exists before proceeding
    await ensureCollection();

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
          id: Date.now(),
          vector: embedding,
          payload: {
            filename: file.name,
          },
        },
      ],
    });

    return NextResponse.json({ success: true });
  } catch (error: unknown) {
    console.error("Error processing image:", error);
    // Error handling...
  }
}
```

This is a Next.js API route that handles image uploads. Here's what happens when a user uploads an image:

1. **Form Data Processing**: 
   - We extract the uploaded file from the form data
   - We check if a file was actually provided

2. **Image Conversion**: 
   - We convert the uploaded file to a base64 string
   - This format makes it easier to process the image with our embedding function

3. **Embedding Generation**: 
   - We call our `generateEmbedding` function to create a vector representation of the image
   - This is where TensorFlow and MobileNet do their magic

4. **Qdrant Storage**: 
   - We use the `upsert` method to add the vector to our Qdrant collection
   - We include the filename as metadata in the "payload" field
   - We use the current timestamp as a unique ID

5. **Response**: 
   - We return a success message if everything worked
   - We handle errors appropriately

This API route acts as a bridge between our frontend and our vector database, processing the image and storing its numerical representation.

## Searching for Similar Images

To find similar images, we generate an embedding for the query image and search for the closest vectors in our Qdrant collection:

```typescript
// app/api/search/route.ts
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
    
    // Generate embedding for the query image
    const embedding = await generateEmbedding(base64Image);
    
    // Search for similar images in Qdrant
    const searchResults = await qdrant.search(COLLECTION_NAME, {
      vector: embedding,
      limit: 5,  // Return top 5 matches
      with_payload: true,
    });
    
    return NextResponse.json({ results: searchResults });
  } catch (error: unknown) {
    // Error handling...
  }
}
```

This search API route follows a similar pattern to our upload route, but with a crucial difference - instead of storing the vector, we use it to search:

1. **Image Processing**: 
   - Just like before, we extract the file and convert it to base64
   - We generate an embedding for this query image

2. **Vector Search**: 
   - We use Qdrant's `search` method to find the most similar vectors
   - We specify a limit of 5 to get the top 5 most similar images
   - We include `with_payload: true` to get the metadata (filename) along with the results

3. **Results**: 
   - Qdrant returns the closest matches along with their similarity scores
   - We pass these results back to the frontend

The magic happens in the `search` method - Qdrant efficiently compares our query vector against all stored vectors using the Cosine distance metric we specified earlier. It returns the closest matches, effectively finding the most visually similar images.

## Building the Frontend

For the frontend, we'll create a simple interface with two main components:

1. An upload form for adding images to our collection
2. A search form for finding similar images

```tsx
// app/page.tsx
'use client';

import { useState } from 'react';
import Image from 'next/image';

export default function Home() {
  const [uploadStatus, setUploadStatus] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);

  async function handleUpload(e) {
    e.preventDefault();
    const formData = new FormData(e.target);
    
    try {
      setUploadStatus('Uploading...');
      const response = await fetch('/api/add-to-collection', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        setUploadStatus('Upload successful!');
      } else {
        setUploadStatus('Upload failed.');
      }
    } catch (error) {
      setUploadStatus('Error uploading image.');
    }
  }

  async function handleSearch(e) {
    e.preventDefault();
    const formData = new FormData(e.target);
    
    try {
      setIsSearching(true);
      const response = await fetch('/api/search', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const data = await response.json();
        setSearchResults(data.results);
      }
    } catch (error) {
      console.error('Error searching:', error);
    } finally {
      setIsSearching(false);
    }
  }

  return (
    <main className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">Image Similarity Search</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="border p-4 rounded">
          <h2 className="text-xl mb-4">Add Images to Collection</h2>
          <form onSubmit={handleUpload}>
            <input type="file" name="image" accept="image/*" required />
            <button type="submit" className="mt-2 px-4 py-2 bg-blue-500 text-white rounded">
              Upload
            </button>
          </form>
          {uploadStatus && <p className="mt-2">{uploadStatus}</p>}
        </div>
        
        <div className="border p-4 rounded">
          <h2 className="text-xl mb-4">Find Similar Images</h2>
          <form onSubmit={handleSearch}>
            <input type="file" name="image" accept="image/*" required />
            <button 
              type="submit" 
              className="mt-2 px-4 py-2 bg-green-500 text-white rounded"
              disabled={isSearching}
            >
              {isSearching ? 'Searching...' : 'Search'}
            </button>
          </form>
          
          {searchResults.length > 0 && (
            <div className="mt-4">
              <h3 className="text-lg font-semibold">Results:</h3>
              <ul className="mt-2">
                {searchResults.map((result, index) => (
                  <li key={index} className="mb-2">
                    <p>Filename: {result.payload.filename}</p>
                    <p>Similarity: {(result.score * 100).toFixed(2)}%</p>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
```

This Next.js page component creates our user interface. Let's break down how it works:

1. **State Management**: 
   - We use React's `useState` hook to manage three pieces of state:
     - `uploadStatus`: Tracks the status of image uploads
     - `searchResults`: Stores the results from our similarity search
     - `isSearching`: Indicates whether a search is in progress

2. **Upload Handler**: 
   - The `handleUpload` function processes the form submission
   - It creates a FormData object from the form
   - It sends a POST request to our `/api/add-to-collection` endpoint
   - It updates the upload status based on the response

3. **Search Handler**: 
   - The `handleSearch` function works similarly
   - It sends the image to our `/api/search` endpoint
   - It updates the `searchResults` state with the response data
   - It manages the loading state with `isSearching`

4. **User Interface**: 
   - We create a two-column layout (on medium screens and larger)
   - The left column contains the upload form
   - The right column contains the search form and results
   - We conditionally render the search results when available
   - We format the similarity score as a percentage

The `'use client'` directive at the top is important - it tells Next.js that this is a client component that uses browser features like state and event handlers.

## Performance Considerations

When implementing vector search, keep these performance considerations in mind:

1. **Model Loading**: TensorFlow.js models can be large. Load them once and reuse the instance.
   - Our singleton pattern in `loadModel()` ensures we only load the model once, saving time and memory.

2. **Image Preprocessing**: Resize and normalize images consistently for accurate embeddings.
   - We resize all images to 224x224 pixels and normalize pixel values to ensure consistent embeddings.

3. **Vector Dimensions**: Higher dimensions capture more features but require more storage and computation.
   - MobileNet's 1280 dimensions strike a good balance between detail and efficiency.

4. **Similarity Metrics**: We use Cosine similarity, but Euclidean or Dot Product might be better for certain applications.
   - Cosine similarity focuses on the direction of vectors rather than their magnitude, which works well for image features.

## Scaling Considerations

As your application grows:

1. **Batch Processing**: Process multiple images in batches for efficiency.
   - Instead of processing one image at a time, you could modify the code to handle multiple uploads simultaneously.

2. **Caching**: Cache embeddings for frequently accessed images.
   - You could add a caching layer (like Redis) to store embeddings for common query images.

3. **Filtering**: Implement pre-filtering based on metadata before vector search.
   - You could add tags or categories to images and filter by these before performing the more expensive vector search.

4. **Hybrid Search**: Combine vector search with traditional filtering for better results.
   - For example, search for "red cars" by combining a text search for "car" with a vector search for the color red.

## Conclusion

Vector search with TensorFlow.js and Qdrant opens up powerful possibilities for content discovery and recommendation systems. This implementation demonstrates how to build a complete image similarity search system using modern web technologies.

The beauty of this approach is its flexibility - you can easily swap MobileNet for other models to extract different features or adapt the system for other content types like text or audio.

By leveraging the power of vector embeddings, we can create intuitive search experiences that understand the semantic meaning of content rather than relying on exact matches or keywords.

Happy coding!

---

_**User**_

what time will it be in new zealand  on Thursday the 20th of March at 5PM PST?

---

_**Assistant**_

# Time Conversion: PST to New Zealand

To convert 5:00 PM PST on Thursday, March 20th to New Zealand time:

1. PST (Pacific Standard Time) is UTC-8
2. New Zealand is currently on NZDT (New Zealand Daylight Time), which is UTC+13
3. The time difference is 21 hours ahead

Therefore, when it's 5:00 PM on Thursday, March 20th in PST, it will be:
- 2:00 PM on Friday, March 21st in New Zealand

Note: If you meant PDT (Pacific Daylight Time) which is UTC-7, then it would be 3:00 PM on Friday, March 21st in New Zealand.

---



---

