import api from "../api";

export const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/api/user/uploads', formData);
    return response.data;
}