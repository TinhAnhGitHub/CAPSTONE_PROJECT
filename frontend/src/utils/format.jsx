export const formatVideoLength = (lengthInSeconds) => {
    const hours = Math.floor(lengthInSeconds / 3600);
    const minutes = Math.floor((lengthInSeconds % 3600) / 60);
    const seconds = Math.floor(lengthInSeconds % 60);
    // show like youtube, if hours > 0, show hh:mm:ss, else show mm:ss, round seconds
    if (hours > 0) {
        return `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    } else {
        return `${minutes}:${String(seconds).padStart(2, '0')}`;
    }
}