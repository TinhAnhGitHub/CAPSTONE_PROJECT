import { GoogleLogin, useGoogleLogin } from '@react-oauth/google'
import React, { useEffect } from 'react'
import { useForm } from "react-hook-form"
import { jwtDecode } from 'jwt-decode';
import api from '@/api/api';
import { useNavigate } from 'react-router-dom';
import { useStore } from '@/stores/user';

export default function Login() {
    const navigate = useNavigate();
    const { login } = useStore.getState();
    const googleLogin = useGoogleLogin({
        flow: 'auth-code',
        onSuccess: async (codeResponse) => {
            const { data } = await api.post(
                'api/user/login/google', {
                code: codeResponse.code,
            });
            const userInfo = data.user;
            login(userInfo, data.access_token);
            navigate('/');
        },
        onError: errorResponse => console.log(errorResponse),
    });

    useEffect(() => {
        // if already logged in, redirect to home
        if (useStore.getState().token) {
            navigate('/');
        }
    }, []);

    return (
        <div className="bg-black/80 h-screen w-screen flex items-center justify-center">
            <div className="bg-white/10 backdrop-blur-lg p-8 rounded-2xl shadow-xl w-full max-w-md">
                <h1 className="text-3xl font-bold text-center text-white mb-6">Sign in</h1>
                <div className="flex justify-center">
                    <button
                        onClick={() => googleLogin()}
                        className="bg-blue-600 text-white px-4 py-2 rounded-lg shadow hover:bg-blue-700"
                    >
                        
                        Sign in with Google
                    </button>
                </div>
            </div>
        </div>
    )

}
