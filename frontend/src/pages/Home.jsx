// Bare-bones ChatGPT-like app for video QA using React + FastAPI + Redis + LlamaIndex + MongoDB + MinIO
// Frontend: React (with React Query, Dropzone, Socket.io)
// Backend: FastAPI, Socket.io, Redis, MongoDB, MinIO, LlamaIndex

// ==== FRONTEND (React) ====
// App.jsx
import Chat from '@/components/Home/Chat';
import Sidebar from '@/components/Home/Sidebar/Sidebar';
import "./gradient.css"
export default function Home() {

    return (
        <div className="flex ">
            <Sidebar />
            <div className="w-full flex justify-center h-screen backdrop-blur-md gradient-background">
                <Chat />
            </div>
        </div>
    );
}
