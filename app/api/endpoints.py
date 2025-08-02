# Member 3: API & Integration Engineer
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from app.retrieval.document_retriever import process_document, search_similar_chunks
from app.llm.answer_generator import generate_answer

router = APIRouter()

class QueryRequest(BaseModel):
    documents: str  # This will contain the document path/ID
    questions: list

@router.post("/run")
async def run_submission(data: QueryRequest):
    try:
        # Use documents field as doc_id
        doc_id = data.documents
        answers = []
        
        for question in data.questions:
            try:
                cleaned_question = question.lower().strip()
                if not cleaned_question:
                    continue
                    
                relevant_chunks = search_similar_chunks(doc_id, cleaned_question)
                answer = await generate_answer(cleaned_question, relevant_chunks)
                answers.append(answer)
            except Exception as e:
                print(f"Error processing question: {e}")
                answers.append(f"Error: {str(e)}")
        
        if not answers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid questions provided"
            )
            
        return {"answers": answers}
    except Exception as e:
        print(f"Error in run_submission: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timed out - processing took too long"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
