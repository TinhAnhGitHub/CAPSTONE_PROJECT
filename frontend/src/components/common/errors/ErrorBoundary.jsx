import { useRouteError } from "react-router-dom";
// import NotFound from '@/pages/errors/NotFound';
// import ServerError from '@/pages/errors/ServerError';

export default function ErrorBoundary() {
    const error = useRouteError();
    console.error("Route error:", error);

    if (error?.status === 404) {
        // return <NotFound />;
        return <div>Page not found</div>
    }

    return <div>Internal server error</div>;
}
