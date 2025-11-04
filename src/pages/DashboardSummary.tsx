import { useEffect, useState } from 'react';
import { api, buildStaticUrl } from '@/lib/api';
import { Layout } from '@/components/Layout';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { BarChart3, Target, TrendingUp, Loader2, AlertCircle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'sonner';

export default function DashboardSummary() {
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
      toast.error('Failed to load summary');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Layout>
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </Layout>
    );
  }

  if (!summary?.hasModel) {
    return (
      <Layout>
        <Card className="p-12 text-center bg-card/50 backdrop-blur-sm">
          <AlertCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <h2 className="text-2xl font-bold mb-2 text-foreground">No Model Trained</h2>
          <p className="text-muted-foreground mb-6">Train your first model to start making predictions</p>
          <Button onClick={() => navigate('/train-models')}>Train Model</Button>
        </Card>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <h2 className="text-3xl font-bold text-foreground mb-2">Model Summary</h2>
          <p className="text-muted-foreground">Performance metrics and configuration</p>
        </div>

        {/* Metrics Grid */}
        {summary.metrics && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50 hover:border-primary/50 transition-all">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2 rounded-lg bg-accent/10">
                  <BarChart3 className="h-5 w-5 text-accent" />
                </div>
                <h3 className="font-semibold text-foreground">Accuracy</h3>
              </div>
              <p className="text-4xl font-bold text-foreground mb-1">
                {((summary.metrics.accuracy || 0) * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-muted-foreground">Classification accuracy</p>
            </Card>

            <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50 hover:border-primary/50 transition-all">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2 rounded-lg bg-accent/10">
                  <Target className="h-5 w-5 text-accent" />
                </div>
                <h3 className="font-semibold text-foreground">RMSE</h3>
              </div>
              <p className="text-4xl font-bold text-foreground mb-1">
                {(summary.metrics.rmse || 0).toFixed(3)}
              </p>
              <p className="text-xs text-muted-foreground">Root mean squared error</p>
            </Card>

            <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50 hover:border-primary/50 transition-all">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2 rounded-lg bg-accent/10">
                  <TrendingUp className="h-5 w-5 text-accent" />
                </div>
                <h3 className="font-semibold text-foreground">R² Score</h3>
              </div>
              <p className="text-4xl font-bold text-foreground mb-1">
                {(summary.metrics.r2 || 0).toFixed(3)}
              </p>
              <p className="text-xs text-muted-foreground">Coefficient of determination</p>
            </Card>
          </div>
        )}

        {/* Configuration */}
        {summary.config && (
          <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
            <h3 className="text-xl font-semibold text-foreground mb-4">Training Configuration</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(summary.config)
                .filter(([key]) => key !== 'GRADE_POINTS')
                .map(([key, value]: any) => (
                  <div key={key} className="p-3 rounded-lg bg-muted/30">
                    <p className="text-xs text-muted-foreground mb-1">{key.replace(/_/g, ' ')}</p>
                    <p className="font-semibold text-foreground">{String(value)}</p>
                  </div>
                ))}
            </div>
          </Card>
        )}

        {/* Plots */}
        {summary.plots && Object.keys(summary.plots).length > 0 && (
          <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
            <h3 className="text-xl font-semibold text-foreground mb-4">Generated Visualizations</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(summary.plots).map(([key, url]: any) => {
                const imageUrl = buildStaticUrl(url);
                return (
                  <div key={key} className="rounded-lg border border-border/40 bg-muted/20 overflow-hidden">
                    <div className="px-4 py-2 border-b border-border/30">
                      <p className="font-medium text-foreground">{key.replace(/_/g, ' ')}</p>
                    </div>
                    <div className="bg-background/80 flex items-center justify-center">
                      <img
                        src={imageUrl || undefined}
                        alt={key}
                        className="w-full h-56 object-contain"
                        loading="lazy"
                      />
                    </div>
                    <p className="px-4 py-2 text-xs text-muted-foreground break-all">{imageUrl || url}</p>
                  </div>
                );
              })}
            </div>
          </Card>
        )}

        {/* Metadata */}
        <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
          <h3 className="text-xl font-semibold text-foreground mb-4">Model Details</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Trained at:</span>
              <span className="text-foreground font-medium">
                {new Date(summary.createdAt).toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Artifacts:</span>
              <span className="text-foreground font-mono text-xs">{summary.artifactsDir}</span>
            </div>
          </div>
        </Card>
      </div>
    </Layout>
  );
}
