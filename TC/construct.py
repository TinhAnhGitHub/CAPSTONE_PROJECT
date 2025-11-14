#!/usr/bin/env python3
import json
from llama_index.core.llms import ChatMessage
from sim import embedding, sim_search

from pydantic import BaseModel
class AgentResponse(BaseModel):
    agent: str
    response : str
class AnswerStorage:
    def __init__(self, filepath: str = "answers.json"):
        """Initialize storage and load or create answers.json."""
        self.filepath = filepath
        self._load_data()

    def _load_data(self):
        """Load data from file."""
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            self.data = []

    def _save_data(self):
        """Save current data to file."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def insert_question(self, question: str):
        """Insert a new question."""
        q_id = len(self.data) + 1
        new_entry = {
            "q_id": q_id,
            "question": question,
            "final": "",
            "stream": []
        }
        self.data.append(new_entry)
        self._save_data()
        print(f"✅ Added question #{q_id}: {question}")

    def append_answer(self, q_id: int, answer: AgentResponse):
        """Append an answer (ChatMessage) to a specific question."""
        q = next((q for q in self.data if q["q_id"] == q_id), None)
        if not q:
            print(f"❌ No question found with id={q_id}")
            return

        # Convert ChatMessage to dict (if it's a Pydantic model)
        if isinstance(answer, AgentResponse):
            q["stream"].append(answer.model_dump())
        else:
            q["stream"].append(answer)

        self._save_data()
        print(f"✅ Appended answer to question #{q_id}")

    def write_final_answer(self, q_id: int, final_text: str):
        """Write the final answer for a question."""
        q = next((q for q in self.data if q["q_id"] == q_id), None)
        if not q:
            print(f"❌ No question found with id={q_id}")
            return

        q["final"] = final_text
        self._save_data()
        print(f"✅ Updated final answer for question #{q_id}")

    def list_questions(self, mode = 0, id = None):
        """Display all stored questions."""
        if not self.data:
            print("No questions yet.")
            return
        print("\n=============== Questions ===============")
        if id is not None:
            q = next((q for q in self.data if q["q_id"] == id), None)
            print(f"ID: {q['q_id']} | Q: {q['question']} ")
            print(f"    Answer: {q['final']}")
            print("========")
            for i, ans in enumerate(q["stream"]):
                print(f"Stream {i+1} ----------------")
                print(f"    Agent: {ans['agent']} | Response: {ans['response']}")

            print("=========================================\n")
            return
        for q in self.data:
            if mode == 0:
                print(f"ID: {q['q_id']} | Q: {q['question']} ")
                print(f"    Answer: {q['final']}")
            if mode == 1 and q["final"] :
                print(f"ID: {q['q_id']} | Q: {q['question']} ")
                print(f"    Answer: {q['final']}")
            elif mode ==1:
                continue
            elif mode == 2 and q["final"]=="":
                print(f"ID: {q['q_id']} | Q: {q['question']} ")
            elif mode == 2:
                continue
            elif mode ==3 and len(q["stream"])>0:
                print(f"ID: {q['q_id']} | Q: {q['question']} ")
                print(f"    Answer: {q['final']}")
                print("========")
            elif mode ==3:
                continue
            elif mode ==4 and len(q["stream"])==0:
                print(f"ID: {q['q_id']} | Q: {q['question']} ")
                print(f"    Answer: {q['final']}")
            elif mode ==4:
                continue


def main():
    store = AnswerStorage("answers.json")
    for q in store.data:
        q["token"] = []
     
    while True:
        print("\n=== MENU ===")
        print("1. Insert question")
        print("2. Append answer")
        print("3. Write final answer")
        print("4. List questions")
        print("5. Test cache")
        print("6. Exit")

        try:
            choice = int(input("Choose option: ").strip())
        except ValueError:
            print("❌ Invalid input.")
            continue

        if choice == 1:
            q = input("Enter question: ").strip()
            store.insert_question(q)
            store._save_data()

        elif choice == 2:

            q_id = int(input("Enter question ID: ").strip())

            store.list_questions(id = q_id)
            agent_name = input("Enter agent name: ").strip()
            ans_text = input("Enter answer text: ").strip()
            doc = {"agent": agent_name, 
                   "response": ans_text}
            if agent_name == "planner":
                doc["reason"] = input("Enter reason for plan: ").strip()
            store.append_answer(q_id, doc)

            store._save_data()

        elif choice == 3:
            q_id = int(input("Enter question ID: ").strip())
            store.list_questions(id = q_id)
            final = input("Enter final answer: ").strip()
            store.write_final_answer(q_id, final)
            store._save_data()

        elif choice == 4:
            try:
                x = int(input("Choose mode: "))
            except ValueError:
                print("❌ Invalid input.")
                continue
            store.list_questions(x)
        elif choice ==5:
            q = input("Write your query:\n")
            res = sim_search(data = store.data, query = q)
            for r in res:
                print(f"Q: {r['question']}\nScore: {r['score']:.3f}\n")
        elif choice == 6:
            print("👋 Exiting...")
            break

        else:
            print("❌ Invalid choice.")

    store._save_data()


if __name__ == "__main__":
    main()
