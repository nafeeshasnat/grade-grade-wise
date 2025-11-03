import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card } from '@/components/ui/card';
import { GradeScaleEditor } from '@/components/GradeScaleEditor';
import { LogStream } from '@/components/LogStream';
import { Upload, Loader2, Sparkles } from 'lucide-react';
import { toast } from 'sonner';

const DEFAULT_CONFIG: any = {
  RANDOM_SEED: 42,
  TEST_SIZE: 0.2,
  THREADS: 4,
  RF_TREES: 300,
  LGBM_N_ESTIMATORS: 2000,
  MLP_HIDDEN: 64,
  MLP_EPOCHS: 300,
  MLP_PATIENCE: 40,
  SVR_ENABLE: true,
  RISK_HIGH_MAX: 3.30,
  RISK_MED_MAX: 3.50,
  GRADE_POINTS: {
    "A+": 4.0, "A": 3.75, "A-": 3.5, "B+": 3.25, "B": 3.0,
    "B-": 2.75, "C+": 2.5, "C": 2.25, "D": 2.0, "F": 0.0
  }
};

export default function TrainModels({ embedded = false }: { embedded?: boolean }) {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [file, setFile] = useState<File | null>(null);
  const [training, setTraining] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const navigate = useNavigate();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleStartTraining = async () => {
    if (!file) {
      toast.error('Please select a training file');
      return;
    }

    setTraining(true);
    try {
      const result = await api.startTraining(file, config);
      setRunId(result.runId);
      toast.success('Training started!');
    } catch (error: any) {
      toast.error(error.message || 'Failed to start training');
      setTraining(false);
    }
  };

  const handleTrainingComplete = (status: string) => {
    if (status === 'SUCCEEDED') {
      toast.success('Training completed successfully!');
      if (embedded) {
        setTimeout(() => navigate('/dashboard/summary'), 1500);
      } else {
        setTimeout(() => navigate('/training-complete'), 1500);
      }
    } else {
      toast.error('Training failed');
      setTraining(false);
      setRunId(null);
    }
  };

  if (runId) {
    const token = localStorage.getItem('token') || '';
    const content = (
      <div className="space-y-6">
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold text-foreground">Training in Progress</h1>
          <p className="text-muted-foreground">Watch your model learn in real-time</p>
        </div>

        <LogStream
          url={api.getTrainLogsUrl(runId)}
          token={token}
          onComplete={handleTrainingComplete}
        />
      </div>
    );

    if (embedded) return content;
    
    return (
      <div className="min-h-screen bg-background p-6">
        <div className="container mx-auto max-w-4xl">
          {content}
        </div>
      </div>
    );
  }

  const content = (
    <Card className="p-8 bg-card/50 backdrop-blur-sm border-border/50 space-y-8">
      {/* Hyperparameters */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold text-foreground">Hyperparameters</h2>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {Object.entries(config)
            .filter(([key]) => !['GRADE_POINTS', 'SVR_ENABLE'].includes(key))
            .map(([key, value]) => (
              <div key={key} className="space-y-2">
                <Label className="text-xs text-muted-foreground">{key.replace(/_/g, ' ')}</Label>
                <Input
                  type="number"
                  value={value as number}
                  onChange={(e) => setConfig({ ...config, [key]: parseFloat(e.target.value) || 0 })}
                  step={key === 'TEST_SIZE' || key.includes('RISK') ? '0.01' : '1'}
                  className="bg-background/50"
                />
              </div>
            ))}
          
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">SVR ENABLE</Label>
            <select
              value={config.SVR_ENABLE ? 'true' : 'false'}
              onChange={(e) => setConfig({ ...config, SVR_ENABLE: e.target.value === 'true' })}
              className="w-full h-10 px-3 rounded-md border border-input bg-background/50 text-sm"
            >
              <option value="true">Yes</option>
              <option value="false">No</option>
            </select>
          </div>
        </div>
      </div>

      {/* Grade Scale */}
      <GradeScaleEditor
        value={config.GRADE_POINTS}
        onChange={(gradePoints) => setConfig({ ...config, GRADE_POINTS: gradePoints })}
      />

      {/* File Upload */}
      <div className="space-y-4">
        <Label className="text-base font-semibold">Training Dataset (JSON)</Label>
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
      </div>

      {/* Submit */}
      <Button
        onClick={handleStartTraining}
        disabled={!file || training}
        size="lg"
        className="w-full gap-2"
      >
        {training && <Loader2 className="h-5 w-5 animate-spin" />}
        Start Training
      </Button>
    </Card>
  );

  if (embedded) return content;

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="container mx-auto max-w-4xl space-y-8">
        <div className="text-center space-y-2 animate-fade-in">
          <h1 className="text-3xl font-bold text-foreground">Configure Training</h1>
          <p className="text-muted-foreground">Set parameters and upload your training dataset</p>
        </div>

        {content}
      </div>
    </div>
  );
}
