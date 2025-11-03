import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { CheckCircle, BarChart3, TrendingUp, Target } from 'lucide-react';
import { toast } from 'sonner';

export default function TrainingComplete() {
  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    loadSummary();
  }, []);

  const loadSummary = async () => {
    try {
      const data = await api.getModelSummary();
      setSummary(data);
    } catch (error: any) {
      toast.error('Failed to load training summary');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary" />
      </div>
    );
  }

  if (!summary?.hasModel) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-6">
        <Card className="p-8 max-w-md text-center bg-card/50 backdrop-blur-sm">
          <p className="text-muted-foreground mb-4">No training results found</p>
          <Button onClick={() => navigate('/train-models')}>Start Training</Button>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="container mx-auto max-w-4xl space-y-8">
        {/* Success Header */}
        <div className="text-center space-y-4 animate-fade-in">
          <div className="inline-flex items-center justify-center p-4 rounded-full bg-primary/10 animate-glow-pulse">
            <CheckCircle className="h-16 w-16 text-primary" />
          </div>
          <h1 className="text-4xl font-bold text-foreground">Training Complete!</h1>
          <p className="text-lg text-muted-foreground">
            Your model has been trained successfully and is ready to make predictions
          </p>
        </div>

        {/* Metrics Cards */}
        {summary.metrics && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 animate-slide-up">
            <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
              <div className="flex items-center gap-3 mb-2">
                <BarChart3 className="h-5 w-5 text-accent" />
                <h3 className="font-semibold text-foreground">Accuracy</h3>
              </div>
              <p className="text-3xl font-bold text-foreground">
                {((summary.metrics.accuracy || 0) * 100).toFixed(1)}%
              </p>
            </Card>

            <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
              <div className="flex items-center gap-3 mb-2">
                <Target className="h-5 w-5 text-accent" />
                <h3 className="font-semibold text-foreground">RMSE</h3>
              </div>
              <p className="text-3xl font-bold text-foreground">
                {(summary.metrics.rmse || 0).toFixed(3)}
              </p>
            </Card>

            <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
              <div className="flex items-center gap-3 mb-2">
                <TrendingUp className="h-5 w-5 text-accent" />
                <h3 className="font-semibold text-foreground">R² Score</h3>
              </div>
              <p className="text-3xl font-bold text-foreground">
                {(summary.metrics.r2 || 0).toFixed(3)}
              </p>
            </Card>
          </div>
        )}

        {/* Plots Available */}
        {summary.plots && Object.keys(summary.plots).length > 0 && (
          <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
            <h3 className="text-lg font-semibold text-foreground mb-4">Available Visualizations</h3>
            <div className="grid grid-cols-2 gap-3">
              {Object.entries(summary.plots).map(([key, path]: any) => (
                <div key={key} className="p-3 rounded-lg bg-muted/50 text-sm">
                  <span className="text-muted-foreground">{key.replace(/_/g, ' ')}</span>
                  <p className="text-xs text-muted-foreground/70 truncate mt-1">{path}</p>
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* Actions */}
        <div className="flex gap-4 justify-center">
          <Button
            size="lg"
            onClick={() => navigate('/dashboard/summary')}
            className="gap-2"
          >
            Go to Dashboard
          </Button>
          <Button
            size="lg"
            variant="outline"
            onClick={() => navigate('/dashboard/predict')}
          >
            Make Prediction
          </Button>
        </div>
      </div>
    </div>
  );
}
