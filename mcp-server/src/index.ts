import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { 
  CallToolRequestSchema, 
  ListToolsRequestSchema,
  Tool 
} from '@modelcontextprotocol/sdk/types.js';
import { fileOperationsTools } from './tools/fileOperations.js';
import { projectTools } from './tools/project.js';
import { codeAnalysisTools } from './tools/codeAnalysis.js';

dotenv.config();

const PORT = process.env.PORT || 3001;
const SERVER_NAME = process.env.MCP_SERVER_NAME || 'cursor-claude-mcp';

class CursorClaudeMCPServer {
  private server: Server;
  private app: express.Application;
  private tools: Tool[] = [];

  constructor() {
    this.server = new Server(
      {
        name: SERVER_NAME,
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.app = express();
    this.setupExpress();
    this.setupMCPHandlers();
    this.registerTools();
  }

  private setupExpress(): void {
    this.app.use(cors({
      origin: process.env.CORS_ORIGINS?.split(',') || '*',
      credentials: true,
    }));
    this.app.use(express.json());
    this.app.use(express.text());

    // Health check endpoint
    this.app.get('/health', (req, res) => {
      res.json({ status: 'ok', server: SERVER_NAME, timestamp: new Date().toISOString() });
    });

    // SSE endpoint for Cursor
    this.app.get('/sse', (req, res) => {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control',
      });

      // Initialize SSE connection
      res.write('data: {"type":"connection","status":"connected"}\n\n');

      // Handle MCP over SSE
      const transport = new StdioServerTransport();
      this.server.connect(transport);

      // Keep connection alive
      const keepAlive = setInterval(() => {
        res.write('data: {"type":"ping"}\n\n');
      }, 30000);

      req.on('close', () => {
        clearInterval(keepAlive);
        res.end();
      });
    });

    // HTTP endpoint for Claude Desktop
    this.app.post('/mcp', async (req, res) => {
      try {
        const response = await this.handleMCPRequest(req.body);
        res.json(response);
      } catch (error) {
        console.error('MCP request error:', error);
        res.status(500).json({ error: 'Internal server error' });
      }
    });
  }

  private setupMCPHandlers(): void {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return { tools: this.tools };
    });

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;
      
      try {
        return await this.executeTool(name, args || {});
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `Error executing tool ${name}: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
          isError: true,
        };
      }
    });
  }

  private registerTools(): void {
    this.tools = [
      ...fileOperationsTools,
      ...projectTools,
      ...codeAnalysisTools,
    ];
  }

  private async executeTool(name: string, args: any): Promise<any> {
    switch (name) {
      case 'read_file':
        return this.executeFileOperation('read', args);
      case 'write_file':
        return this.executeFileOperation('write', args);
      case 'list_files':
        return this.executeFileOperation('list', args);
      case 'search_files':
        return this.executeFileOperation('search', args);
      case 'get_project_structure':
        return this.executeProjectOperation('structure', args);
      case 'analyze_code':
        return this.executeCodeAnalysis('analyze', args);
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  }

  private async executeFileOperation(operation: string, args: any): Promise<any> {
    const { fileOperations } = await import('./handlers/fileOperations.js');
    return (fileOperations as any)[operation](args);
  }

  private async executeProjectOperation(operation: string, args: any): Promise<any> {
    const { projectOperations } = await import('./handlers/project.js');
    return (projectOperations as any)[operation](args);
  }

  private async executeCodeAnalysis(operation: string, args: any): Promise<any> {
    const { codeAnalysis } = await import('./handlers/codeAnalysis.js');
    return (codeAnalysis as any)[operation](args);
  }

  private async handleMCPRequest(body: any): Promise<any> {
    // Handle direct HTTP MCP requests for Claude Desktop
    const transport = new StdioServerTransport();
    return new Promise((resolve, reject) => {
      this.server.connect(transport);
      // Process the request and return response
      resolve({ status: 'processed', data: body });
    });
  }

  public async start(): Promise<void> {
    this.app.listen(PORT, () => {
      console.log(`🚀 MCP Server running on port ${PORT}`);
      console.log(`📡 SSE endpoint: http://localhost:${PORT}/sse`);
      console.log(`🔗 HTTP endpoint: http://localhost:${PORT}/mcp`);
      console.log(`❤️  Health check: http://localhost:${PORT}/health`);
    });
  }
}

// Start the server
const server = new CursorClaudeMCPServer();
server.start().catch(console.error);

export default CursorClaudeMCPServer;