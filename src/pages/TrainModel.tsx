import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Brain, Zap, Target, TrendingUp, ArrowRight } from 'lucide-react';

export default function TrainModel() {
  const navigate = useNavigate();

  const features = [
    {
      icon: Brain,
      title: 'Advanced ML Models',
      description: 'Ensemble of Random Forest, LightGBM, MLP, and SVR'
    },
    {
      icon: Target,
      title: 'High Accuracy',
      description: 'Optimized hyperparameters for best prediction performance'
    },
    {
      icon: TrendingUp,
      title: 'Risk Analysis',
      description: 'Identify at-risk students before it\'s too late'
    },
    {
      icon: Zap,
      title: 'Real-time Training',
      description: 'Stream logs and monitor progress as your model trains'
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="relative overflow-hidden py-20 px-6">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-accent/5 to-background pointer-events-none" />
        
        <div className="container mx-auto max-w-4xl relative">
          <div className="text-center space-y-8 animate-fade-in">
            <div className="inline-flex items-center justify-center p-6 rounded-3xl bg-primary/10 backdrop-blur-sm animate-glow-pulse">
              <Brain className="h-16 w-16 text-primary" />
            </div>
            
            <div className="space-y-4">
              <h1 className="text-5xl md:text-6xl font-bold text-foreground">
                Train Your First Model
              </h1>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                Upload your training data and let our ML system learn patterns to predict academic performance
              </p>
            </div>

            <Button
              size="lg"
              onClick={() => navigate('/train-models')}
              className="gap-2 text-lg px-8 py-6 animate-slide-up"
            >
              Start Training
              <ArrowRight className="h-5 w-5" />
            </Button>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-16 px-6">
        <div className="container mx-auto max-w-6xl">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <Card
                  key={index}
                  className="p-6 bg-card/50 backdrop-blur-sm border-border/50 hover:border-primary/50 transition-all duration-300 hover:shadow-lg hover:shadow-primary/10 animate-slide-up"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <div className="flex items-start gap-4">
                    <div className="p-3 rounded-lg bg-primary/10">
                      <Icon className="h-6 w-6 text-primary" />
                    </div>
                    <div className="space-y-1">
                      <h3 className="text-lg font-semibold text-foreground">{feature.title}</h3>
                      <p className="text-sm text-muted-foreground">{feature.description}</p>
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>
        </div>
      </section>

      {/* How it Works */}
      <section className="py-16 px-6">
        <div className="container mx-auto max-w-4xl">
          <h2 className="text-3xl font-bold text-center mb-12 text-foreground">How It Works</h2>
          
          <div className="space-y-6">
            {[
              { step: 1, title: 'Configure Parameters', desc: 'Set hyperparameters or use our optimized defaults' },
              { step: 2, title: 'Upload Training Data', desc: 'Provide historical student data in JSON format' },
              { step: 3, title: 'Monitor Training', desc: 'Watch real-time logs as models train' },
              { step: 4, title: 'Review Metrics', desc: 'Analyze performance metrics and visualizations' }
            ].map((item) => (
              <Card key={item.step} className="p-6 bg-card/30 backdrop-blur-sm border-border/30">
                <div className="flex items-center gap-4">
                  <div className="flex items-center justify-center w-12 h-12 rounded-full bg-primary text-primary-foreground font-bold text-lg">
                    {item.step}
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-foreground">{item.title}</h3>
                    <p className="text-sm text-muted-foreground">{item.desc}</p>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
