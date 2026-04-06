import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { QueryClient, QueryClientProvider } from 'react-query'
import router from './routes';
import { RouterProvider } from 'react-router-dom'
import { GoogleOAuthProvider } from '@react-oauth/google'
import { googleClientId } from './constants/auth'

const queryClient = new QueryClient()
createRoot(document.getElementById('root')).render(
  // <StrictMode>
    <QueryClientProvider client={queryClient}>
      <GoogleOAuthProvider clientId={googleClientId}>
        <RouterProvider router={router}>
          <App />
        </RouterProvider>
      </GoogleOAuthProvider>
    </QueryClientProvider>
  // </StrictMode>
)
