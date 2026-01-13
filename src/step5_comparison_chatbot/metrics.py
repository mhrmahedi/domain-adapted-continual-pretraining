class ResponseMetrics:
    def calculate_metrics(self, response, generation_time):
        words = response.split()
        return {
            "word_count": len(words),
            "generation_time": round(generation_time, 3),
            "tokens_per_second": round(len(words) / generation_time, 2) if generation_time > 0 else 0
        }
    
    def compare_responses(self, metrics_base, metrics_adapted):
        if metrics_base["generation_time"] < metrics_adapted["generation_time"]:
            faster = "Base"
            speedup = metrics_adapted["generation_time"] / metrics_base["generation_time"]
        else:
            faster = "Adapted"
            speedup = metrics_base["generation_time"] / metrics_adapted["generation_time"]
        
        comparison = f"""⚡ **Speed**: {faster} model is {speedup:.2f}x faster
  **Length**: Base has {metrics_base['word_count']} words, Adapted has {metrics_adapted['word_count']} words"""
        
        return comparison
