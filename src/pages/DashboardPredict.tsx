import { useState } from 'react';
import { api } from '@/lib/api';
import { Layout } from '@/components/Layout';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Upload, Loader2, Zap } from 'lucide-react';
import { toast } from 'sonner';

export default function DashboardPredict() {
  const [file, setFile] = useState<File | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null);
    }
  };

  const handlePredict = async () => {
    if (!file) {
      toast.error('Please select a student file');
      return;
    }

    setPredicting(true);
    try {
      const data = await api.predict(file);
      setResult(data);
      toast.success('Prediction completed!');
    } catch (error: any) {
      toast.error(error.message || 'Prediction failed');
    } finally {
      setPredicting(false);
    }
  };

  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <h2 className="text-3xl font-bold text-foreground mb-2">Make Prediction</h2>
          <p className="text-muted-foreground">Upload student data to predict academic performance</p>
        </div>

        {/* Upload Card */}
        <Card className="p-8 bg-card/50 backdrop-blur-sm border-border/50">
          <div className="space-y-6">
            <div className="space-y-4">
              <Label className="text-base font-semibold">Student Data (JSON)</Label>
              <div className="flex gap-4">
                <Input
                  type="file"
                  accept=".json,application/json"
                  onChange={handleFileChange}
                  className="flex-1 bg-background/50"
                />
                {file && (
                  <div className="flex items-center gap-2 px-4 py-2 rounded-md bg-primary/10 text-primary text-sm">
                    <Upload className="h-4 w-4" />
                    {file.name}
                  </div>
                )}
              </div>
              <p className="text-xs text-muted-foreground">
                Upload a JSON file containing student information for prediction
              </p>
            </div>

            <Button
              onClick={handlePredict}
              disabled={!file || predicting}
              size="lg"
              className="w-full gap-2"
            >
              {predicting ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Zap className="h-5 w-5" />
              )}
              {predicting ? 'Predicting...' : 'Run Prediction'}
            </Button>
          </div>
        </Card>

        {/* Results */}
        {result && (
          <Card className="p-8 bg-card/50 backdrop-blur-sm border-border/50 animate-fade-in">
            <h3 className="text-2xl font-bold text-foreground mb-6">Prediction Results</h3>
            
            <div className="space-y-4">
              <div className="p-4 rounded-lg bg-primary/10 border border-primary/20">
                <p className="text-sm text-muted-foreground mb-1">Student ID</p>
                <p className="text-xl font-bold text-foreground">{result.studentId}</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {result.results && Object.entries(result.results).map(([key, value]: any) => (
                  <div key={key} className="p-4 rounded-lg bg-muted/30">
                    <p className="text-sm text-muted-foreground mb-1">{key.replace(/_/g, ' ')}</p>
                    <p className="text-lg font-semibold text-foreground">
                      {typeof value === 'number' ? value.toFixed(3) : String(value)}
                    </p>
                  </div>
                ))}
              </div>

              {result.results?.predicted_grade && (
                <div className="p-6 rounded-lg bg-gradient-to-br from-primary/20 to-accent/20 border border-primary/30">
                  <p className="text-sm text-muted-foreground mb-2">Predicted Grade</p>
                  <p className="text-5xl font-bold text-foreground">{result.results.predicted_grade}</p>
                </div>
              )}

              {result.results?.risk_level && (
                <div className={`p-4 rounded-lg border ${
                  result.results.risk_level === 'high' ? 'bg-destructive/10 border-destructive/30' :
                  result.results.risk_level === 'medium' ? 'bg-yellow-500/10 border-yellow-500/30' :
                  'bg-green-500/10 border-green-500/30'
                }`}>
                  <p className="text-sm text-muted-foreground mb-1">Risk Level</p>
                  <p className="text-lg font-bold capitalize">{result.results.risk_level}</p>
                </div>
              )}

              <div className="pt-4 border-t border-border/50">
                <p className="text-xs text-muted-foreground">
                  Predicted at: {new Date(result.createdAt).toLocaleString()}
                </p>
              </div>
            </div>
          </Card>
        )}
      </div>
    </Layout>
  );
}
