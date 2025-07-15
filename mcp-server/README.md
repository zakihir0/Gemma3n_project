# Cursor-Claude MCP Server

リモートMCPサーバーでClaude DesktopとCursorを連携させるためのサーバーです。

## 特徴

- 🔗 Claude DesktopとCursor両方に対応
- 📁 ファイル操作機能（読み取り、書き込み、検索）
- 🏗️ プロジェクト構造分析
- 🔍 コード解析機能
- 🌐 SSE（Server-Sent Events）とHTTP両方サポート
- 🔒 OAuth認証対応（Claude Desktop用）

## 必要環境

- Node.js 18.0.0以上
- npm または yarn

## インストール

### 自動インストール

**macOS/Linux:**
```bash
./start-server.sh
```

**Windows:**
```cmd
install.bat
```

### 手動インストール

1. 依存関係をインストール:
```bash
npm install
```

2. 環境設定ファイルを作成:
```bash
cp .env.example .env
```

3. プロジェクトをビルド:
```bash
npm run build
```

4. サーバーを起動:
```bash
npm start
```

## 設定

### Cursor用設定

Cursorの設定ファイル (`~/.cursor/mcp_servers.json`) に以下を追加:

```json
{
  "mcpServers": {
    "cursor-claude-mcp": {
      "transport": "sse",
      "url": "http://localhost:3001/sse",
      "env": {
        "TRANSPORT": "sse"
      }
    }
  }
}
```

### Claude Desktop用設定

Claude Desktopで以下の手順で設定:

1. Settings > Integrations を開く
2. "Add Integration" をクリック
3. Server URL: `http://localhost:3001/mcp`
4. 認証方法を選択（OAuth推奨）

## 利用可能なツール

### ファイル操作
- `read_file` - ファイル内容の読み取り
- `write_file` - ファイルへの書き込み
- `list_files` - ディレクトリ内のファイル一覧
- `search_files` - ファイル内テキスト検索

### プロジェクト管理
- `get_project_structure` - プロジェクト構造の取得
- `get_project_info` - プロジェクト情報の取得
- `find_config_files` - 設定ファイルの検索

### コード解析
- `analyze_code` - コード構造分析
- `get_function_definitions` - 関数定義の抽出
- `find_imports` - インポート文の検索
- `check_syntax` - 構文チェック

## 開発

開発モードで起動:
```bash
npm run dev
```

テストの実行:
```bash
npm test
```

## エンドポイント

- **Health Check**: `GET http://localhost:3001/health`
- **SSE (Cursor用)**: `GET http://localhost:3001/sse`
- **HTTP (Claude Desktop用)**: `POST http://localhost:3001/mcp`

## トラブルシューティング

### 接続できない場合

1. サーバーが起動しているか確認:
   ```bash
   curl http://localhost:3001/health
   ```

2. ポートが使用されていないか確認:
   ```bash
   lsof -i :3001
   ```

3. ファイアウォール設定を確認

### Cursorで認識されない場合

1. Cursorを再起動
2. 設定ファイルのパスを確認
3. JSON形式が正しいか確認

### Claude Desktopで接続エラーが発生する場合

1. OAuth認証の設定を確認
2. CORSの設定を確認
3. ネットワーク接続を確認

## ライセンス

MIT License