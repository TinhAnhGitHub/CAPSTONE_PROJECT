import { PRIMARY_URL } from '@/constants/url';
import { io } from "socket.io-client";
import toast from 'react-hot-toast';
const socket = io(PRIMARY_URL);

socket.on('connect', () => {
    toast.success('Connected to agent!', {
        className: 'animate-[var(--animate-highlight-pulse)]',
        style: {
            background: 'var(--color-surface)',
            color: 'var(--color-text)',
            border: '1px solid var(--color-accent-muted)',
        },
        iconTheme: {
            primary: 'var(--color-accent)',
            secondary: 'var(--color-background)',
        },
    });
});
 
export default socket;