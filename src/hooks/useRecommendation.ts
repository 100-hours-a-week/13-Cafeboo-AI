import { useMutation } from '@tanstack/react-query';
import { CaffeineProfile, getCaffeineLimit, getCoffeeRecommendation } from '../services/api';

export const useRecommendation = () => {
  const caffeineLimitMutation = useMutation({
    mutationFn: (profile: CaffeineProfile) => getCaffeineLimit(profile),
  });

  const coffeeRecommendationMutation = useMutation({
    mutationFn: (profile: CaffeineProfile) => getCoffeeRecommendation(profile),
  });

  return {
    getCaffeineLimit: caffeineLimitMutation.mutate,
    getCoffeeRecommendation: coffeeRecommendationMutation.mutate,
    isLoadingLimit: caffeineLimitMutation.isPending,
    isLoadingRecommendation: coffeeRecommendationMutation.isPending,
    limitData: caffeineLimitMutation.data,
    recommendationData: coffeeRecommendationMutation.data,
    limitError: caffeineLimitMutation.error,
    recommendationError: coffeeRecommendationMutation.error,
  };
}; 