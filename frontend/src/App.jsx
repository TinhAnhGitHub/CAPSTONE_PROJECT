import React from 'react'
import Home from './pages/Home'
import { Outlet } from 'react-router-dom'
import  { Toaster } from 'react-hot-toast';

export default function App() {
  return (
    <div>
      <Outlet />
      <Toaster reverseOrder={false} />
    </div>
  )
}
