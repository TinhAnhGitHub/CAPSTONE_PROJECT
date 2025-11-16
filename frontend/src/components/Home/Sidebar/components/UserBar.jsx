import { useStore } from '@/stores/user';

import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom';
import LibraryModal from './Library/LibraryModal';

export default function UserBar() {
    const user = useStore((state) => state.user)
    const logout = useStore.getState().logout;
    const logoutConfirm = () => {
        if (confirm("Are you sure you want to logout?")) {
            logout();
        }
    }
    const navigate = useNavigate();

    const [isModalOpen, setIsModalOpen] = useState(false);
    const closeModal = () => setIsModalOpen(false);
    const openModal = () => setIsModalOpen(true);


    return (
        <div className='my-auto'>
            {/* user img*/}
            <div className='flex items-center gap-2'>
                {user ? <img src={user?.picture} alt={user?.name} className='w-8 h-8 rounded-full' /> :
                    <div className='w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center text-white'>
                        U
                    </div>
                }
                <div className='flex-1'>
                    {user ? user.name : 'Guest'}
                </div>
                
                <button className='bg-gray-800 px-2 py-1 rounded hover:bg-gray-700 cursor-pointer transition'
                    onClick={() => user ? logoutConfirm() : navigate('/login')}
                >
                    {user ? 'Logout' : 'Login'}
                </button>

            </div>
        </div>
    )
}
