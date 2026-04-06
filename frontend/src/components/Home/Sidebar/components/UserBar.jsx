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
            <div className='flex items-center gap-3'>
                {user ? <img src={user?.picture} alt={user?.name} className='w-9 h-9 rounded-full' /> :
                    <div className='w-9 h-9 rounded-full bg-surface-light flex items-center justify-center text-text-muted font-medium'>
                        G
                    </div>
                }
                <div className='flex-1 text-text'>
                    {user ? user.name : 'Guest'}
                </div>

                <button className='bg-surface-light hover:bg-surface-hover text-text-muted hover:text-text px-3 py-1.5 rounded-lg text-sm font-medium cursor-pointer transition-colors'
                    onClick={() => user ? logoutConfirm() : navigate('/login')}
                >
                    {user ? 'Logout' : 'Login'}
                </button>

            </div>
        </div>
    )
}
