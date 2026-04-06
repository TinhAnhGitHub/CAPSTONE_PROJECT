import api from '@/api/api';
import React from 'react'
import { useQuery } from 'react-query'

export default function PrivateHome() {
    const { data } = useQuery('userData', async () => {
        const response = await api.get('/api/login/secure-endpoint')
        return response.data
    })
    return (
        <div>
            Private Home
            {JSON.stringify(data)}
        </div>
    )
}
