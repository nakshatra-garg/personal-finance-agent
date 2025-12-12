"""RAG (Retrieval Augmented Generation) system for document Q&A."""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from llm_client import LLMClient


class RAGSystem:
    """Simple RAG system for document question answering."""

    def __init__(self, llm_client: LLMClient, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG system.

        Args:
            llm_client: LLM client for generation
            embedding_model: Sentence transformer model name
        """
        self.llm_client = llm_client
        self.embedder = SentenceTransformer(embedding_model)
        self.documents: List[Dict[str, Any]] = []
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_document(self, text: str, metadata: Dict[str, Any]):
        """
        Add a document to the RAG system.

        Args:
            text: Document text
            metadata: Document metadata (filename, type, etc.)
        """
        # Split text into chunks
        chunks = self._chunk_text(text, chunk_size=500, overlap=50)

        # Store document
        doc_id = len(self.documents)
        self.documents.append({
            "id": doc_id,
            "text": text,
            "metadata": metadata,
            "chunks": chunks
        })

        # Add chunks to corpus
        for chunk in chunks:
            self.chunks.append(chunk)

        # Update embeddings
        self._update_embeddings()

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind(".")
                last_newline = chunk.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > chunk_size * 0.5:  # Only if reasonable
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())
            start = end - overlap

        return [c for c in chunks if c]  # Remove empty chunks

    def _update_embeddings(self):
        """Update embeddings for all chunks."""
        if self.chunks:
            self.embeddings = self.embedder.encode(
                self.chunks,
                convert_to_numpy=True,
                show_progress_bar=False
            )

    def query(
        self,
        question: str,
        top_k: int = 3,
        include_sources: bool = True
    ) -> str:
        """
        Answer a question using RAG.

        Args:
            question: User question
            top_k: Number of relevant chunks to retrieve
            include_sources: Whether to include source citations

        Returns:
            Generated answer
        """
        if not self.chunks or self.embeddings is None:
            return "No documents have been uploaded yet. Please upload financial documents first."

        # Get relevant chunks
        relevant_chunks = self._retrieve_relevant_chunks(question, top_k)

        if not relevant_chunks:
            return "I couldn't find relevant information in the uploaded documents to answer your question."

        # Generate answer
        answer = self._generate_answer(question, relevant_chunks)

        if include_sources and relevant_chunks:
            answer += "\n\n**Sources:**\n"
            for i, chunk in enumerate(relevant_chunks[:2], 1):
                preview = chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]
                answer += f"{i}. {preview}\n"

        return answer

    def _retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks for a query.

        Args:
            query: Search query
            top_k: Number of chunks to retrieve

        Returns:
            List of relevant chunks with metadata
        """
        # Encode query
        query_embedding = self.embedder.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return relevant chunks
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Threshold for relevance
                results.append({
                    "text": self.chunks[idx],
                    "score": float(similarities[idx])
                })

        return results

    def _generate_answer(
        self,
        question: str,
        relevant_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate answer using LLM with retrieved context.

        Args:
            question: User question
            relevant_chunks: Retrieved relevant chunks

        Returns:
            Generated answer
        """
        # Build context from chunks
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])

        prompt = f"""You are a helpful financial assistant. Answer the question based on the provided context from financial documents.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer based on the context provided
- Be specific and cite numbers/details from the documents
- If the context doesn't contain enough information, say so
- Keep the answer concise and focused

Answer:"""

        messages = [
            {
                "role": "system",
                "content": "You are a helpful financial assistant that answers questions based on provided documents."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            return self.llm_client.get_response_text(response)

        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def get_document_summary(self) -> str:
        """
        Get summary of all uploaded documents.

        Returns:
            Summary text
        """
        if not self.documents:
            return "No documents uploaded yet."

        summary_parts = [f"**Uploaded Documents: {len(self.documents)}**\n"]

        for doc in self.documents:
            metadata = doc["metadata"]
            summary_parts.append(
                f"- {metadata.get('name', 'Unknown')}: "
                f"{len(doc['chunks'])} sections, "
                f"{len(doc['text'])} characters"
            )

        return "\n".join(summary_parts)

    def clear(self):
        """Clear all documents and embeddings."""
        self.documents = []
        self.chunks = []
        self.embeddings = None
