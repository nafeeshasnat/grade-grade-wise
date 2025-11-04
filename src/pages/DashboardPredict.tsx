import { useState } from 'react';
import { api } from '@/lib/api';
import { Layout } from '@/components/Layout';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Upload, Loader2, Zap } from 'lucide-react';
import { toast } from 'sonner';
import PredictionDetails from '@/components/predictions/PredictionDetails';

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
            <PredictionDetails prediction={result} />
          </Card>
        )}
      </div>
    </Layout>
  );
}
