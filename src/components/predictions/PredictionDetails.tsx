import { buildStaticUrl } from '@/lib/api';

type PredictionRecord = {
  id?: string;
  studentId?: string;
  createdAt?: string;
  results?: Record<string, any>;
  summary?: Record<string, any>;
  outFile?: string | null;
  outFileUrl?: string | null;
  inputPath?: string | null;
  plots?: Record<string, string>;
};

interface PredictionDetailsProps {
  prediction: PredictionRecord;
}

const formatValue = (value: any) =>
  typeof value === 'number'
    ? value.toFixed(3)
    : typeof value === 'undefined' || value === null
    ? '-'
    : String(value);

const toEntries = (obj: Record<string, any>) =>
  Object.entries(obj || {}).filter(([_, value]) => value !== undefined && value !== null);

export function PredictionDetails({ prediction }: PredictionDetailsProps) {
  const summary = prediction.summary || {};
  const predictions = prediction.results || {};

  const finalCgpa = predictions.final_cgpa || {};
  const nextSem = predictions.next_sem_gpa || {};
  const ensemble = summary.ensemble || predictions.ensemble || {};
  const riskLevel = summary.risk || 'Unknown';
  const current = summary.current || {};
  const bestModel = summary.bestModel || prediction.bestModel;

  const files = summary.files || {};
  const inputUrl = buildStaticUrl(files.input || prediction.inputFileUrl || prediction.inputPath || null);
  const outputUrl = buildStaticUrl(files.output || prediction.outFileUrl || prediction.outFile || null);

  const plotSources = prediction.plots || summary.plots || {};

  const riskTone =
    riskLevel?.toLowerCase() === 'high'
      ? 'bg-destructive/10 border-destructive/30'
      : riskLevel?.toLowerCase() === 'medium'
      ? 'bg-yellow-500/10 border-yellow-500/30'
      : 'bg-green-500/10 border-green-500/30';

  return (
    <div className="space-y-6">
      <div className="p-4 rounded-lg bg-primary/10 border border-primary/20">
        <p className="text-sm text-muted-foreground mb-1">Student ID</p>
        <p className="text-xl font-bold text-foreground">{prediction.studentId || 'Unknown'}</p>
      </div>

      <div className={`p-4 rounded-lg border ${riskTone}`}>
        <p className="text-sm text-muted-foreground mb-1">Risk Level</p>
        <p className="text-lg font-bold capitalize">{riskLevel}</p>
      </div>

      {bestModel && (
        <div className="p-3 rounded-lg bg-accent/10 border border-accent/30">
          <p className="text-xs text-muted-foreground mb-1">Best Performing Model (from training)</p>
          <p className="text-sm font-semibold text-foreground">{String(bestModel)}</p>
        </div>
      )}

      {(current.last_sem_gpa || current.current_cgpa) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 rounded-lg bg-muted/30">
            <p className="text-sm text-muted-foreground mb-1">Latest Semester GPA</p>
            <p className="text-lg font-semibold text-foreground">{formatValue(current.last_sem_gpa)}</p>
            {current.last_sem_index && (
              <p className="text-xs text-muted-foreground mt-1">Semester {current.last_sem_index}</p>
            )}
          </div>
          <div className="p-4 rounded-lg bg-muted/30">
            <p className="text-sm text-muted-foreground mb-1">Current CGPA</p>
            <p className="text-lg font-semibold text-foreground">{formatValue(current.current_cgpa)}</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 rounded-lg bg-muted/20 border border-border/40">
          <p className="text-sm font-semibold text-foreground mb-3">Final CGPA by Model</p>
          <div className="space-y-2">
            {toEntries(finalCgpa).map(([model, value]) => {
              const highlight = bestModel && String(bestModel).toLowerCase() === model.toLowerCase();
              return (
                <div
                  key={model}
                  className={`flex justify-between text-sm rounded-md px-3 py-2 ${
                    highlight ? 'bg-primary/10 border border-primary/40' : 'bg-background/40'
                  }`}
                >
                  <span className="text-muted-foreground">
                    {model} {highlight && <span className="text-primary text-xs font-semibold">(Best)</span>}
                  </span>
                  <span className="font-medium text-foreground">{formatValue(value)}</span>
                </div>
              );
            })}
            {toEntries(finalCgpa).length === 0 && (
              <p className="text-xs text-muted-foreground">No final CGPA predictions available.</p>
            )}
          </div>
        </div>

        <div className="p-4 rounded-lg bg-muted/20 border border-border/40">
          <p className="text-sm font-semibold text-foreground mb-3">Next Semester GPA by Model</p>
          <div className="space-y-2">
            {toEntries(nextSem).map(([model, value]) => (
              <div key={model} className="flex justify-between text-sm rounded-md px-3 py-2 bg-background/40">
                <span className="text-muted-foreground">{model}</span>
                <span className="font-medium text-foreground">{formatValue(value)}</span>
              </div>
            ))}
            {toEntries(nextSem).length === 0 && (
              <p className="text-xs text-muted-foreground">No next semester predictions available.</p>
            )}
          </div>
        </div>
      </div>

      {(ensemble.final_cgpa_mean || ensemble.next_sem_gpa_mean) && (
        <div className="p-6 rounded-lg bg-gradient-to-br from-primary/20 to-accent/20 border border-primary/30">
          <p className="text-sm text-muted-foreground mb-2">Ensemble Averages</p>
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
            <div>
              <p className="text-xs text-muted-foreground">Final CGPA (Mean)</p>
              <p className="text-3xl font-bold text-foreground">{formatValue(ensemble.final_cgpa_mean)}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Next Sem GPA (Mean)</p>
              <p className="text-3xl font-bold text-foreground">{formatValue(ensemble.next_sem_gpa_mean)}</p>
            </div>
          </div>
        </div>
      )}

      {Object.keys(plotSources).length > 0 && (
        <div className="space-y-3">
          <h4 className="text-lg font-semibold text-foreground">Prediction Visualizations</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(plotSources).map(([key, url]) => {
              const resolved = buildStaticUrl(url);
              return (
                <div key={key} className="rounded-lg border border-border/40 bg-muted/20 overflow-hidden">
                  <div className="px-3 py-2 border-b border-border/30">
                    <span className="text-sm font-medium text-foreground">{key.replace(/_/g, ' ')}</span>
                  </div>
                  <div className="bg-background/80 flex items-center justify-center">
                    <img
                      src={resolved || undefined}
                      alt={key}
                      className="w-full h-52 object-contain"
                      loading="lazy"
                    />
                  </div>
                  <p className="px-3 py-2 text-[11px] text-muted-foreground break-all">{resolved || url}</p>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="pt-4 border-t border-border/50 text-xs text-muted-foreground space-y-2">
        <p>Predicted at: {prediction.createdAt ? new Date(prediction.createdAt).toLocaleString() : 'Unknown'}</p>
        {inputUrl && (
          <p>
            Uploaded file:{' '}
            <a href={inputUrl} target="_blank" rel="noreferrer" className="text-primary underline">
              Download
            </a>
          </p>
        )}
        {outputUrl && (
          <p>
            Detailed output:{' '}
            <a href={outputUrl} target="_blank" rel="noreferrer" className="text-primary underline">
              Download
            </a>
          </p>
        )}
        {prediction.id && (
          <p>
            Prediction ID: <span className="font-mono break-all text-foreground">{prediction.id}</span>
          </p>
        )}
      </div>
    </div>
  );
}

export default PredictionDetails;
