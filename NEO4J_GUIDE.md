# Neo4j Graph Database Explorer Guide

## 🎯 What You're Looking At

Your Neo4j database contains the **knowledge graph** of your Walmart retail data, visualized as an interactive network of nodes and relationships.

## 📊 Visual Overview

### All Nodes View
![Neo4j All Nodes](file:///C:/Users/swara/.gemini/antigravity/brain/3f223ef2-7099-4abc-ab18-f3cd32d411e4/neo4j_all_nodes_1763659208783.png)

This shows 25 **Store nodes** from your database. Each circle represents a Walmart store with properties like:
- `store_id`: Unique identifier
- `type`: Store type (A, B, or C)
- `size`: Store size in square feet

### Store Relationships View
![Neo4j Store Relationships](file:///C:/Users/swara/.gemini/antigravity/brain/3f223ef2-7099-4abc-ab18-f3cd32d411e4/neo4j_store_relationships_v2_1763659249480.png)

This visualization shows **SIMILAR_TO** relationships between stores. The arrows connect stores that are similar based on type and performance characteristics.

## 🔍 Exploring Your Data

### Basic Queries to Try

#### 1. View All Stores
```cypher
MATCH (s:Store)
RETURN s
LIMIT 25
```
**What it does**: Shows first 25 stores in your database

#### 2. Find Similar Stores
```cypher
MATCH (s:Store {store_id: 1})-[:SIMILAR_TO]->(similar:Store)
RETURN s, similar
```
**What it does**: Shows stores similar to Store 1

#### 3. View Store Details
```cypher
MATCH (s:Store {store_id: 1})
RETURN s.store_id, s.type, s.size
```
**What it does**: Shows properties of Store 1

#### 4. Find All Type A Stores
```cypher
MATCH (s:Store {type: 'A'})
RETURN s
LIMIT 10
```
**What it does**: Shows first 10 Type A stores

#### 5. Count Stores by Type
```cypher
MATCH (s:Store)
RETURN s.type, COUNT(s) as count
ORDER BY count DESC
```
**What it does**: Counts how many stores of each type

#### 6. Find Departments for a Store
```cypher
MATCH (s:Store {store_id: 1})-[:SELLS_IN]->(d:Department)
RETURN s, d
LIMIT 10
```
**What it does**: Shows departments in Store 1

#### 7. Complex Pattern: Store Network
```cypher
MATCH path = (s1:Store)-[:SIMILAR_TO*1..2]-(s2:Store)
WHERE s1.store_id = 1
RETURN path
LIMIT 20
```
**What it does**: Shows stores similar to Store 1 and their similar stores (2 degrees)

## 🎨 Visualization Tips

### In Neo4j Browser:

1. **Click on nodes** to see their properties
2. **Double-click nodes** to expand relationships
3. **Drag nodes** to rearrange the graph
4. **Use mouse wheel** to zoom in/out
5. **Click relationships** to see relationship properties

### View Modes:

- **Graph View**: Visual network (default)
- **Table View**: Data in rows/columns
- **Text View**: Raw JSON response
- **Code View**: Shows the Cypher query

## 📈 Understanding Your Graph

### Node Types in Your Database:

1. **Store** (45 nodes)
   - Properties: `store_id`, `type`, `size`
   - Represents physical Walmart stores

2. **Department** (many nodes)
   - Properties: `dept_id`, `name`, `avg_sales`
   - Represents product departments

### Relationship Types:

1. **SIMILAR_TO**
   - Connects stores with similar characteristics
   - Used for recommendations and comparisons

2. **SELLS_IN**
   - Connects stores to departments
   - Shows which departments exist in each store

3. **HAS_FEATURE**
   - Connects stores to temporal features
   - Links to holiday flags, economic indicators

## 🚀 Advanced Queries

### Find Top Performing Stores
```cypher
MATCH (s:Store)-[:SELLS_IN]->(d:Department)
WITH s, AVG(d.avg_sales) as avg_dept_sales
RETURN s.store_id, s.type, avg_dept_sales
ORDER BY avg_dept_sales DESC
LIMIT 10
```

### Store Similarity Network
```cypher
MATCH (s:Store)-[r:SIMILAR_TO]->(similar:Store)
WHERE s.type = 'A'
RETURN s, r, similar
LIMIT 20
```

### Department Distribution
```cypher
MATCH (s:Store)-[:SELLS_IN]->(d:Department)
RETURN s.store_id, COUNT(d) as dept_count
ORDER BY dept_count DESC
LIMIT 10
```

## 🔗 How This Powers Your Platform

### 1. RAG Pipeline
When you ask: *"Which stores are similar to Store 1?"*

Neo4j executes:
```cypher
MATCH (s:Store {store_id: 1})-[:SIMILAR_TO]->(similar:Store)
RETURN similar
```

Then feeds this context to GPT for a natural language answer.

### 2. Graph Insights Endpoint
`POST /graph-insights` uses Neo4j to:
- Get store information
- Find similar stores
- Retrieve departments
- Provide context for forecasts

### 3. Recommendation System
The GNN (Graph Neural Network) uses Neo4j's structure to:
- Learn store embeddings
- Predict revenue influence
- Find hidden patterns

## 📝 Quick Reference

### Access Neo4j:
- **Browser UI**: http://localhost:7474
- **Username**: neo4j
- **Password**: password
- **Bolt Port**: 7687 (for API)

### Useful Commands:
```cypher
// Show database schema
CALL db.schema.visualization()

// Count all nodes
MATCH (n) RETURN COUNT(n)

// Count all relationships
MATCH ()-[r]->() RETURN COUNT(r)

// Show node labels
CALL db.labels()

// Show relationship types
CALL db.relationshipTypes()
```

## 🎓 Learning Resources

### Cypher Query Language:
- **MATCH**: Find patterns in the graph
- **WHERE**: Filter results
- **RETURN**: Specify what to return
- **CREATE**: Add new nodes/relationships
- **DELETE**: Remove nodes/relationships
- **SET**: Update properties

### Pattern Syntax:
- `(n)`: Any node
- `(n:Label)`: Node with label
- `(n {prop: value})`: Node with property
- `-[r]->`: Directed relationship
- `-[r:TYPE]->`: Typed relationship
- `-[*1..3]->`: Variable length path (1-3 hops)

## 🛠️ Troubleshooting

### Can't Connect?
```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Restart Neo4j
docker-compose restart neo4j
```

### Empty Database?
```bash
# Reload data
python main.py --steps ingest
```

### Slow Queries?
```cypher
// Create indexes for better performance
CREATE INDEX FOR (s:Store) ON (s.store_id)
CREATE INDEX FOR (d:Department) ON (d.dept_id)
```

## 🎯 Next Steps

1. **Explore the visualizations** - Click around and see the connections
2. **Try the example queries** - Copy/paste into Neo4j Browser
3. **Ask questions via web UI** - Use your chatbox at http://localhost:3000
4. **Check API integration** - See how Neo4j powers `/graph-insights`

Your graph database is the **knowledge backbone** of your AI platform! 🚀
