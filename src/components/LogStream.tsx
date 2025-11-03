import { useEffect, useRef, useState } from 'react';
import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Terminal } from 'lucide-react';

interface LogStreamProps {
  url: string;
  token: string;
  onComplete?: (status: string) => void;
}

export function LogStream({ url, token, onComplete }: LogStreamProps) {
  const [logs, setLogs] = useState<string>('');
  const [status, setStatus] = useState<'connecting' | 'streaming' | 'complete' | 'error'>('connecting');
  const scrollRef = useRef<HTMLPreElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const eventSource = new EventSource(`${url}?token=${encodeURIComponent(token)}`);

    eventSource.onopen = () => {
      setStatus('streaming');
    };

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.content) {
          setLogs((prev) => prev + data.content);
        }

        if (data.complete) {
          setStatus('complete');
          onComplete?.(data.status);
          eventSource.close();
        }
      } catch (error) {
        console.error('Failed to parse log event:', error);
      }
    };

    eventSource.onerror = () => {
      setStatus('error');
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [url, token, onComplete]);

  useEffect(() => {
    // Auto-scroll to bottom
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <Card className="bg-background/95 backdrop-blur-sm border-border/50 overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border/50 bg-muted/30">
        <Terminal className="h-4 w-4 text-accent" />
        <span className="text-sm font-medium text-foreground">Training Logs</span>
        <div className="ml-auto flex items-center gap-2">
          <div className={`h-2 w-2 rounded-full ${
            status === 'streaming' ? 'bg-accent animate-pulse' : 
            status === 'complete' ? 'bg-primary' : 
            status === 'error' ? 'bg-destructive' : 
            'bg-muted-foreground'
          }`} />
          <span className="text-xs text-muted-foreground capitalize">{status}</span>
        </div>
      </div>
      
      <div ref={containerRef} className="h-[400px] overflow-y-auto">
        <pre 
          ref={scrollRef}
          className="p-4 text-xs font-mono text-foreground/90 whitespace-pre-wrap break-words"
        >
          {logs || 'Waiting for logs...'}
        </pre>
      </div>
    </Card>
  );
}
