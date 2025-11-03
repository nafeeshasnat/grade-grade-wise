import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { Layout } from '@/components/Layout';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, Eye, History } from 'lucide-react';
import { toast } from 'sonner';

export default function DashboardHistory() {
  const [predictions, setPredictions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPrediction, setSelectedPrediction] = useState<any>(null);

  useEffect(() => {
    loadPredictions();
  }, []);

  const loadPredictions = async () => {
    try {
      const data = await api.getPredictions();
      setPredictions(data);
    } catch (error: any) {
      toast.error('Failed to load predictions');
    } finally {
      setLoading(false);
    }
  };

  const handleViewDetails = async (id: string) => {
    try {
      const data = await api.getPrediction(id);
      setSelectedPrediction(data);
    } catch (error: any) {
      toast.error('Failed to load prediction details');
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

  if (selectedPrediction) {
    return (
      <Layout>
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-3xl font-bold text-foreground">Prediction Details</h2>
            <Button variant="outline" onClick={() => setSelectedPrediction(null)}>
              Back to List
            </Button>
          </div>

          <Card className="p-8 bg-card/50 backdrop-blur-sm border-border/50">
            <div className="space-y-6">
              <div className="p-4 rounded-lg bg-primary/10 border border-primary/20">
                <p className="text-sm text-muted-foreground mb-1">Student ID</p>
                <p className="text-2xl font-bold text-foreground">{selectedPrediction.studentId}</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {selectedPrediction.results && Object.entries(selectedPrediction.results).map(([key, value]: any) => (
                  <div key={key} className="p-4 rounded-lg bg-muted/30">
                    <p className="text-sm text-muted-foreground mb-1">{key.replace(/_/g, ' ')}</p>
                    <p className="text-lg font-semibold text-foreground">
                      {typeof value === 'number' ? value.toFixed(3) : String(value)}
                    </p>
                  </div>
                ))}
              </div>

              <div className="pt-4 border-t border-border/50 space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Prediction ID:</span>
                  <span className="text-foreground font-mono">{selectedPrediction.id}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Created at:</span>
                  <span className="text-foreground">{new Date(selectedPrediction.createdAt).toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Input file:</span>
                  <span className="text-foreground font-mono text-xs">{selectedPrediction.inputPath}</span>
                </div>
              </div>
            </div>
          </Card>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <h2 className="text-3xl font-bold text-foreground mb-2">Prediction History</h2>
          <p className="text-muted-foreground">View all past predictions</p>
        </div>

        {predictions.length === 0 ? (
          <Card className="p-12 text-center bg-card/50 backdrop-blur-sm">
            <History className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-foreground mb-2">No Predictions Yet</h3>
            <p className="text-muted-foreground">Make your first prediction to see history</p>
          </Card>
        ) : (
          <div className="space-y-3">
            {predictions.map((pred) => (
              <Card
                key={pred.id}
                className="p-6 bg-card/50 backdrop-blur-sm border-border/50 hover:border-primary/50 transition-all"
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <p className="text-lg font-semibold text-foreground">
                        Student: {pred.studentId}
                      </p>
                      {pred.results?.predicted_grade && (
                        <span className="px-3 py-1 rounded-full bg-primary/10 text-primary text-sm font-medium">
                          {pred.results.predicted_grade}
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {new Date(pred.createdAt).toLocaleString()}
                    </p>
                  </div>
                  
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleViewDetails(pred.id)}
                    className="gap-2"
                  >
                    <Eye className="h-4 w-4" />
                    View
                  </Button>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </Layout>
  );
}
