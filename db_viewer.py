import sqlite3
from datetime import datetime

def view_database():
    conn = sqlite3.connect("/Users/devshah/Documents/WorkSpace/University/year 3/CSC393/empathic-AI-agent/database_temp.db")
    cur = conn.cursor()
    
    print("\n=== Chat Sessions ===")
    print("Session ID | Created At")
    print("-" * 50)
    
    cur.execute("SELECT session_id, created_at FROM chat_sessions")
    sessions = cur.fetchall()
    for session in sessions:
        print(f"{session[0]} | {session[1]}")
    
    print("\n=== Messages ===")
    print("ID | Session ID | Role | Content")
    print("-" * 100)
    
    cur.execute("SELECT id, session_id, role, content FROM messages")
    messages = cur.fetchall()
    for msg in messages:
        print(f"{msg[0]} | {msg[1]} | {msg[2]} | {msg[3]}")
    
    conn.close()

if __name__ == "__main__":
    view_database()