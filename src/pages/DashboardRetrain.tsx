import TrainModels from './TrainModels';

export default function DashboardRetrain() {
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-3xl font-bold text-foreground mb-2">Retrain Model</h2>
        <p className="text-muted-foreground">
          Train a new model version with updated data or different parameters
        </p>
      </div>

      {/* Use TrainModels in embedded mode */}
      <div className="mt-6">
        <TrainModels embedded={true} />
      </div>
    </div>
  );
}
