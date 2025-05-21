import React from 'react';
import { useForm } from 'react-hook-form';
import { useRecommendation } from '../hooks/useRecommendation';
import { CaffeineProfile } from '../services/api';

export const CaffeineForm = () => {
  const { register, handleSubmit } = useForm<CaffeineProfile>();
  const {
    getCaffeineLimit,
    getCoffeeRecommendation,
    isLoadingLimit,
    isLoadingRecommendation,
    limitData,
    recommendationData,
    limitError,
    recommendationError,
  } = useRecommendation();

  const onSubmit = (data: CaffeineProfile) => {
    getCaffeineLimit(data);
    getCoffeeRecommendation(data);
  };

  return (
    <div className="p-4">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">성별 (1: 남성, 2: 여성)</label>
          <input
            type="number"
            {...register('gender', { required: true, min: 1, max: 2 })}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700">나이</label>
          <input
            type="number"
            {...register('age', { required: true, min: 0 })}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">키 (cm)</label>
          <input
            type="number"
            step="0.1"
            {...register('height', { required: true, min: 0 })}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">체중 (kg)</label>
          <input
            type="number"
            step="0.1"
            {...register('weight', { required: true, min: 0 })}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">흡연 여부 (0: 비흡연, 1: 흡연)</label>
          <input
            type="number"
            {...register('is_smoker', { required: true, min: 0, max: 1 })}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">호르몬 피임약 복용 (0: 미복용, 1: 복용)</label>
          <input
            type="number"
            {...register('take_hormonal_contraceptive', { required: true, min: 0, max: 1 })}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">카페인 민감도 (0-100)</label>
          <input
            type="number"
            {...register('caffeine_sensitivity', { required: true, min: 0, max: 100 })}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">현재 카페인 섭취량 (mg)</label>
          <input
            type="number"
            {...register('current_caffeine', { required: true, min: 0 })}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">카페인 제한량 (mg)</label>
          <input
            type="number"
            {...register('caffeine_limit', { required: true, min: 0 })}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">수면 시 잔여 카페인 (mg)</label>
          <input
            type="number"
            step="0.1"
            {...register('residual_at_sleep', { required: true, min: 0 })}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">목표 수면 시 잔여 카페인 (mg)</label>
          <input
            type="number"
            step="0.1"
            {...register('target_residual_at_sleep', { required: true, min: 0 })}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">계획된 카페인 섭취량 (mg)</label>
          <input
            type="number"
            {...register('planned_caffeine_intake', { required: true, min: 0 })}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">현재 시간 (24시간 형식)</label>
          <input
            type="number"
            step="0.1"
            {...register('current_time', { required: true, min: 0, max: 24 })}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">취침 시간 (24시간 형식)</label>
          <input
            type="number"
            step="0.1"
            {...register('sleep_time', { required: true, min: 0, max: 24 })}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
          />
        </div>

        <button
          type="submit"
          className="w-full bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600"
          disabled={isLoadingLimit || isLoadingRecommendation}
        >
          {isLoadingLimit || isLoadingRecommendation ? '로딩 중...' : '분석하기'}
        </button>
      </form>

      {limitData && (
        <div className="mt-4 p-4 bg-green-50 rounded-md">
          <h3 className="text-lg font-medium text-green-800">카페인 제한 분석 결과</h3>
          <pre className="mt-2 text-sm text-green-700">{JSON.stringify(limitData, null, 2)}</pre>
        </div>
      )}

      {recommendationData && (
        <div className="mt-4 p-4 bg-blue-50 rounded-md">
          <h3 className="text-lg font-medium text-blue-800">커피 추천 결과</h3>
          <pre className="mt-2 text-sm text-blue-700">{JSON.stringify(recommendationData, null, 2)}</pre>
        </div>
      )}

      {(limitError || recommendationError) && (
        <div className="mt-4 p-4 bg-red-50 rounded-md">
          <h3 className="text-lg font-medium text-red-800">에러 발생</h3>
          <p className="mt-2 text-sm text-red-700">
            {limitError?.message || recommendationError?.message}
          </p>
        </div>
      )}
    </div>
  );
}; 