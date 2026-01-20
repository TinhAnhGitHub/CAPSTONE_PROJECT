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
        onError: errorResponse => console.error(errorResponse),
    });

    useEffect(() => {
        // if already logged in, redirect to home
        if (useStore.getState().token) {
            navigate('/');
        }
    }, []);

    return (
        <div className="bg-background h-screen w-screen flex items-center justify-center">
            <div className="bg-surface border border-surface-light p-8 rounded-2xl shadow-xl w-full max-w-md">
                <h1 className="text-3xl font-bold text-center text-text mb-6">Sign in</h1>
                <div className="flex justify-center">
                    <button
                        onClick={() => googleLogin()}
                        className="flex items-center gap-3 bg-white text-gray-700 px-6 py-3 rounded-lg shadow-md hover:shadow-lg hover:bg-gray-50 transition-all cursor-pointer font-medium"
                    >
                        <svg className="w-5 h-5" viewBox="0 0 24 24">
                            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
                        </svg>
                        Continue with Google
                    </button>
                </div>
            </div>
        </div>
    )

}
