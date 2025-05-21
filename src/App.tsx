import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { CaffeineForm } from './components/CaffeineForm';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-8">카페인 분석 및 커피 추천</h1>
        <CaffeineForm />
      </div>
    </QueryClientProvider>
  );
}

export default App; 