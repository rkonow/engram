//! SQL over the tables of one database (DESIGN.md §3.7): `CREATE TABLE` and
//! `DROP TABLE`, queries and writes on any table by name, references
//! between tables, and the models of each table's embedded fields.

use std::sync::Arc;

use strata::db::{Db, SqlResult};
use strata::embed::{Embedder, HashEmbedder};
use strata::query::Params;
use strata::schema::{Field, FieldType, Schema};
use strata::storage::{CheckpointOptions, DbOptions, Layout};
use strata::value::Value;

fn papers() -> Schema {
    Schema::new(
        "papers",
        vec![
            Field::new("id", FieldType::Keyword),
            Field::new("title", FieldType::Text { positions: true }),
            Field::new("year", FieldType::Int),
            Field::new("author", FieldType::Keyword),
        ],
        Some("id"),
    )
    .unwrap()
}

fn run(db: &Db, sql: &str) -> SqlResult {
    db.sql(sql, &Params::new()).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn written(db: &Db, sql: &str) -> u64 {
    match run(db, sql) {
        SqlResult::Written(n) => n,
        SqlResult::Rows(_) => panic!("{sql}: expected a write"),
    }
}

fn rows(db: &Db, sql: &str) -> Vec<Vec<Value>> {
    match run(db, sql) {
        SqlResult::Rows(r) => r.rows,
        SqlResult::Written(_) => panic!("{sql}: expected rows"),
    }
}

/// The error `sql` fails with.
fn fails(db: &Db, sql: &str) -> String {
    match db.sql(sql, &Params::new()) {
        Ok(_) => panic!("{sql}: expected an error"),
        Err(e) => e.to_string(),
    }
}

fn s(v: &str) -> Value {
    Value::Str(v.into())
}

fn names(db: &Db) -> Vec<String> {
    db.catalog().iter().map(|t| t.table.clone()).collect()
}

#[test]
fn tables_through_sql_single_file() {
    tables_through_sql(Layout::SingleFile);
}

#[test]
fn tables_through_sql_directory() {
    tables_through_sql(Layout::Directory);
}

fn tables_through_sql(layout: Layout) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.strata");
    let opts = || DbOptions { layout, sync: false, ..DbOptions::default() };
    let db = Db::create(&path, papers(), opts()).unwrap();
    assert_eq!(written(&db, "CREATE TABLE authors (id KEYWORD PRIMARY KEY, name TEXT, born INT, topics KEYWORD[])"), 0);
    assert!(fails(&db, "CREATE TABLE Authors (id KEYWORD PRIMARY KEY)").contains("already exists"));
    // IF NOT EXISTS leaves the table there as it is.
    assert_eq!(written(&db, "CREATE TABLE IF NOT EXISTS authors (id INT PRIMARY KEY)"), 0);
    assert_eq!(names(&db), ["papers", "authors"]);
    assert_eq!(db.storage().table("authors").unwrap().fields.len(), 4);

    assert_eq!(
        written(
            &db,
            "INSERT INTO authors (id, name, born, topics) VALUES
               ('ada', 'Ada Lovelace', 1815, ARRAY['math', 'engines']),
               ('alan', 'Alan Turing', 1912, ARRAY['math', 'computing']),
               ('grace', 'Grace Hopper', 1906, ARRAY['computing', 'compilers'])"
        ),
        3
    );
    assert_eq!(written(&db, "INSERT INTO papers (id, title, year, author) VALUES ('p1', 'computable numbers', 1936, 'alan'), ('p2', 'notes on the analytical engine', 1843, 'ada')"), 2);

    // Each query reads the table it names, in any case.
    assert_eq!(rows(&db, "SELECT id FROM authors WHERE topics = 'computing' ORDER BY born"), [[s("grace")], [s("alan")]]);
    assert_eq!(rows(&db, "SELECT id, score() FROM AUTHORS WHERE MATCH(name, 'turing')")[0][0], s("alan"));
    assert_eq!(rows(&db, "SELECT COUNT(*) FROM papers"), [[Value::Int(2)]]);
    assert_eq!(rows(&db, "SELECT id FROM papers WHERE MATCH(title, 'engine')"), [[s("p2")]]);
    // Compound queries too, on the same snapshot.
    assert_eq!(rows(&db, "WITH old AS (SELECT id FROM authors WHERE born < 1910) SELECT COUNT(*) FROM old"), [[Value::Int(2)]]);
    assert_eq!(rows(&db, "SELECT id FROM authors WHERE id IN (SELECT id FROM authors WHERE topics = 'math') ORDER BY born DESC"), [[s("alan")], [s("ada")]]);
    assert_eq!(rows(&db, "WITH authors AS (SELECT id, born FROM authors WHERE born > 1900) SELECT COUNT(*) FROM authors"), [[Value::Int(2)]]);

    // And across tables (tests/joins_across_tables.rs has more).
    assert_eq!(rows(&db, "SELECT id FROM papers WHERE author IN (SELECT id FROM authors WHERE born > 1900)"), [[s("p1")]]);
    assert_eq!(rows(&db, "SELECT p.id, a.name FROM papers p JOIN authors a ON p.author = a.id ORDER BY p.id"), [[s("p1"), s("Alan Turing")], [s("p2"), s("Ada Lovelace")]]);
    assert_eq!(rows(&db, "WITH a AS (SELECT id FROM authors) SELECT id FROM papers WHERE author IN (SELECT id FROM a) ORDER BY year"), [[s("p2")], [s("p1")]]);
    assert!(fails(&db, "SELECT id FROM nope").contains("no such table: nope"));

    // Writes go to the table they name.
    assert_eq!(written(&db, "UPDATE authors SET born = born + 1 WHERE id = 'ada'"), 1);
    assert_eq!(written(&db, "UPDATE Authors SET name = 'Grace Brewster Hopper' WHERE id = 'grace'"), 1);
    assert_eq!(rows(&db, "SELECT born FROM authors WHERE MATCH(name, 'brewster')"), [[Value::Int(1906)]]);
    assert_eq!(written(&db, "INSERT INTO authors (id, name, born) VALUES ('ada', 'x', 0), ('kurt', 'Kurt Godel', 1906) ON CONFLICT (id) DO NOTHING"), 1);
    assert_eq!(written(&db, "INSERT INTO authors (id, name, born) VALUES ('kurt', 'Kurt Goedel', 1906) ON CONFLICT (id) DO UPDATE SET name = excluded.name"), 1);
    assert_eq!(rows(&db, "SELECT id FROM authors WHERE MATCH(name, 'goedel')"), [[s("kurt")]]);
    assert_eq!(written(&db, "DELETE FROM authors WHERE born > 1900 ORDER BY born DESC LIMIT 1"), 1);
    assert_eq!(rows(&db, "SELECT id FROM authors ORDER BY born, id"), [[s("ada")], [s("grace")], [s("kurt")]]);
    assert!(fails(&db, "INSERT INTO nope (id) VALUES ('x')").contains("no such table: nope"));
    // The first table is untouched.
    assert_eq!(rows(&db, "SELECT id, author FROM papers ORDER BY year"), [[s("p2"), s("ada")], [s("p1"), s("alan")]]);

    // The table and its rows outlive a reopen.
    drop(db);
    let db = Db::open(&path, opts()).unwrap();
    assert_eq!(names(&db), ["papers", "authors"]);
    assert_eq!(rows(&db, "SELECT id, born FROM authors WHERE born < 1910 ORDER BY born, id"), [[s("ada"), Value::Int(1816)], [s("grace"), Value::Int(1906)], [s("kurt"), Value::Int(1906)]]);

    // DROP TABLE: not the first table; IF EXISTS.
    assert!(fails(&db, "DROP TABLE papers").contains("cannot be dropped"));
    assert_eq!(written(&db, "DROP TABLE authors"), 0);
    assert!(fails(&db, "SELECT id FROM authors").contains("no such table: authors"));
    assert!(fails(&db, "DROP TABLE authors").contains("no table authors"));
    assert_eq!(written(&db, "DROP TABLE IF EXISTS authors"), 0);
    assert_eq!(names(&db), ["papers"]);
    assert_eq!(rows(&db, "SELECT COUNT(*) FROM papers"), [[Value::Int(2)]]);
    // A table created again starts empty.
    assert_eq!(written(&db, "CREATE TABLE authors (id INT PRIMARY KEY, name TEXT)"), 0);
    assert_eq!(rows(&db, "SELECT COUNT(*) FROM authors"), [[Value::Int(0)]]);
    assert!(fails(&db, "INSERT INTO authors (id, name) VALUES ('ada', 'Ada')").contains("does not fit"));
}

/// A quoted table name is the name inside the quotes, wherever a statement
/// names a table: `"papers"` reads papers. Before, the quotes stayed in the
/// name and every quoted table was "no such table".
#[test]
fn quoted_table_names_name_the_table() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::create(&dir.path().join("t.strata"), papers(), DbOptions { sync: false, ..DbOptions::default() }).unwrap();
    assert_eq!(written(&db, r#"CREATE TABLE "authors" (id KEYWORD PRIMARY KEY, name TEXT, born INT)"#), 0);
    assert_eq!(names(&db), ["papers", "authors"]);
    assert_eq!(written(&db, r#"INSERT INTO "authors" (id, name, born) VALUES ('ada', 'Ada Lovelace', 1815), ('alan', 'Alan Turing', 1912)"#), 2);
    assert_eq!(written(&db, r#"INSERT INTO "papers" (id, title, year, author) VALUES ('p1', 'computable numbers', 1936, 'alan'), ('p2', 'the analytical engine', 1843, 'ada')"#), 2);

    assert_eq!(rows(&db, r#"SELECT id FROM "papers" WHERE year > 1900"#), [[s("p1")]]);
    assert_eq!(rows(&db, r#"SELECT id FROM "Authors" WHERE MATCH(name, 'turing')"#), [[s("alan")]]);
    assert_eq!(rows(&db, r#"SELECT p.id FROM "papers" AS p WHERE p.year < 1900"#), [[s("p2")]]);
    // Compound queries: WITH (quoted result names too), subqueries, joins
    // and their qualified wildcards.
    assert_eq!(rows(&db, r#"WITH "old" AS (SELECT id FROM "authors" WHERE born < 1900) SELECT COUNT(*) FROM "old""#), [[Value::Int(1)]]);
    assert_eq!(rows(&db, r#"SELECT id FROM "papers" WHERE year IN (SELECT year FROM "papers" WHERE year > 1900)"#), [[s("p1")]]);
    assert_eq!(
        rows(&db, r#"SELECT "a".id, b.id FROM "papers" "a" JOIN "papers" b ON "a".author = b.author WHERE "a".year < 1900"#),
        [[s("p2"), s("p2")]]
    );
    assert_eq!(rows(&db, r#"SELECT "a".* FROM "papers" "a" JOIN papers b ON "a".id = b.id WHERE b.year > 1900"#), [[s("p1"), s("computable numbers"), Value::Int(1936), s("alan")]]);
    assert_eq!(rows(&db, r#"SELECT DISTINCT author FROM "papers" ORDER BY author"#), [[s("ada")], [s("alan")]]);
    assert!(!rows(&db, r#"EXPLAIN SELECT id FROM "papers" WHERE year > 1900"#).is_empty());

    assert_eq!(written(&db, r#"UPDATE "authors" SET born = 1816 WHERE id = 'ada'"#), 1);
    assert_eq!(written(&db, r#"DELETE FROM "authors" WHERE id = 'alan'"#), 1);
    assert_eq!(rows(&db, "SELECT id, born FROM authors"), [[s("ada"), Value::Int(1816)]]);
    assert!(fails(&db, r#"SELECT id FROM "nope""#).contains("no such table: nope"));
    assert_eq!(written(&db, r#"DROP TABLE "authors""#), 0);
    assert_eq!(names(&db), ["papers"]);
}

#[test]
fn references_are_recorded_and_kept_consistent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("r.strata");
    let db = Db::create(&path, papers(), DbOptions::default()).unwrap();
    written(&db, "CREATE TABLE authors (id KEYWORD PRIMARY KEY, name TEXT)");
    written(&db, "CREATE TABLE log (line TEXT)");
    written(&db, "CREATE TABLE wrote (id INT PRIMARY KEY, author KEYWORD REFERENCES AUTHORS, paper KEYWORD, next INT[] REFERENCES wrote, FOREIGN KEY (paper) REFERENCES papers (id))");
    let wrote = db.storage().table("wrote").unwrap();
    let refs: Vec<Option<&str>> = wrote.fields.iter().map(|f| f.references.as_deref()).collect();
    // Spelled as the table is.
    assert_eq!(refs, [None, Some("authors"), Some("papers"), Some("wrote")]);

    for (bad, why) in [
        ("CREATE TABLE t (a KEYWORD REFERENCES nope)", "references nope, which is not a table"),
        ("CREATE TABLE t (a INT REFERENCES authors)", "whose primary key id is KEYWORD, not INT"),
        ("CREATE TABLE t (a KEYWORD REFERENCES authors (name))", "a reference names its table's primary key, which is id"),
        ("CREATE TABLE t (a KEYWORD REFERENCES log)", "references log, which has no primary key"),
        ("CREATE TABLE t (a TEXT REFERENCES authors)", "only a KEYWORD, INT or EDGES field references a table"),
        ("CREATE TABLE t (id KEYWORD PRIMARY KEY, a EDGES REFERENCES authors)", "an EDGES field links rows of its own table"),
    ] {
        let e = fails(&db, bad);
        assert!(e.contains(why), "{bad}: {e}");
    }
    assert!(db.storage().table("t").is_none());
    // A new database has only its own table to reference.
    let first = Schema::new("p", vec![Field::new("id", FieldType::Keyword), Field::new("a", FieldType::Keyword).references("authors")], Some("id")).unwrap();
    assert!(Db::create(&dir.path().join("other.strata"), first, DbOptions::default()).is_err());

    // Recorded, not enforced.
    assert_eq!(written(&db, "INSERT INTO wrote (id, author, paper, next) VALUES (1, 'nobody', 'p9', ARRAY[7, 8])"), 1);

    // A referenced table goes with the tables that reference it, or with
    // CASCADE, which removes the references.
    let e = fails(&db, "DROP TABLE authors");
    assert!(e.contains("wrote.author references authors"), "{e}");
    assert!(fails(&db, "DROP TABLE authors RESTRICT").contains("wrote.author references authors"));
    assert_eq!(written(&db, "DROP TABLE authors CASCADE"), 0);
    let refs = |db: &Db| db.storage().table("wrote").unwrap().fields.iter().map(|f| f.references.clone()).collect::<Vec<_>>();
    assert_eq!(refs(&db), [None, None, Some("papers".to_string()), Some("wrote".to_string())]);
    assert_eq!(rows(&db, "SELECT author FROM wrote"), [[s("nobody")]]);
    drop(db);
    let db = Db::open(&path, DbOptions::default()).unwrap();
    assert_eq!(refs(&db), [None, None, Some("papers".to_string()), Some("wrote".to_string())]);

    written(&db, "CREATE TABLE notes (id INT PRIMARY KEY, about INT REFERENCES wrote)");
    // A checkpoint creates the tables again in their order, in which a
    // table comes after those it references, and keeps the references.
    let copy = dir.path().join("copy.strata");
    db.storage().checkpoint(&copy, &CheckpointOptions::default()).unwrap();
    let c = Db::open(&copy, DbOptions::default()).unwrap();
    assert_eq!(names(&c), names(&db));
    assert_eq!(refs(&c), refs(&db));
    assert_eq!(c.storage().table("notes").unwrap().fields[1].references.as_deref(), Some("wrote"));
    drop(c);
    assert!(fails(&db, "DROP TABLE wrote").contains("notes.about references wrote"));
    // Dropped together, and a table's references to itself do not hold it.
    assert_eq!(written(&db, "DROP TABLE wrote, notes"), 0);
    assert_eq!(names(&db), ["papers", "log"]);
    // The first table is never dropped, so references to it never block.
    written(&db, "CREATE TABLE cites (id INT PRIMARY KEY, paper KEYWORD REFERENCES papers)");
    assert_eq!(written(&db, "DROP TABLE cites, log"), 0);
}

#[test]
fn each_table_embeds_its_own_fields() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("e.strata");
    let models = dir.path().join("models");
    std::fs::create_dir(&models).unwrap();
    let db = Db::create(&path, papers(), DbOptions::default()).unwrap();
    assert!(db.open_embedders(&models).is_empty());
    // A table created after open_embedders opens its models from there.
    written(&db, "CREATE TABLE notes (id KEYWORD PRIMARY KEY, body TEXT, emb VECTOR OPTIONS(embed = 'body', model = 'hash:16', index = 'flat'))");
    assert!(db.embedder_in("notes", "emb").is_some());
    assert!(db.embedder("emb").is_none(), "the first table has no such field");
    let h = HashEmbedder { dim: 16 };
    let vector = |text: &str| Value::Vector(h.embed(&[text]).unwrap().remove(0).into());
    written(&db, "INSERT INTO notes (id, body) VALUES ('n1', 'graph walks over citations'), ('n2', 'vector search with codes')");
    assert_eq!(rows(&db, "SELECT emb FROM notes WHERE id = 'n1'"), [[vector("graph walks over citations")]]);
    let near = |db: &Db, text: &str| {
        let p = Params::new().set("q", vector(text));
        match db.sql("SELECT id FROM notes ORDER BY similarity(emb, :q) DESC LIMIT 1", &p).unwrap() {
            SqlResult::Rows(r) => r.rows[0][0].clone(),
            SqlResult::Written(_) => unreachable!(),
        }
    };
    assert_eq!(near(&db, "vector search with codes"), s("n2"));

    // Rewriting a source recomputes the vector, by UPDATE or by ON CONFLICT DO UPDATE.
    written(&db, "UPDATE notes SET body = 'binary codes for vectors' WHERE id = 'n1'");
    assert_eq!(rows(&db, "SELECT emb FROM notes WHERE id = 'n1'"), [[vector("binary codes for vectors")]]);
    written(&db, "INSERT INTO notes (id, body) VALUES ('n2', 'random walks on graphs') ON CONFLICT (id) DO UPDATE SET body = excluded.body");
    assert_eq!(rows(&db, "SELECT emb FROM notes WHERE id = 'n2'"), [[vector("random walks on graphs")]]);
    db.load_into("notes", [vec![s("n3"), s("a third note"), Value::Null]]).unwrap();
    assert_eq!(rows(&db, "SELECT emb FROM notes WHERE id = 'n3'"), [[vector("a third note")]]);

    // A model is set for a table's field, with that field's dimensions.
    assert!(db.set_embedder_in("notes", "emb", Arc::new(HashEmbedder { dim: 8 })).is_err());
    assert!(db.set_embedder_in("notes", "body", Arc::new(HashEmbedder { dim: 16 })).is_err());
    assert!(db.set_embedder_in("nope", "emb", Arc::new(HashEmbedder { dim: 16 })).is_err());

    // A model that does not open leaves the vectors to the writes, and is
    // named with its table.
    written(&db, "CREATE TABLE other (id INT PRIMARY KEY, t TEXT, v VECTOR(4) OPTIONS(embed = 't', model = 'no-such-model'))");
    assert!(db.embedder_in("other", "v").is_none());
    assert_eq!(written(&db, "INSERT INTO other (id, t) VALUES (1, 'x')"), 1);
    drop(db);
    let db = Db::open(&path, DbOptions::default()).unwrap();
    let failed: Vec<String> = db.open_embedders(&models).into_iter().map(|(f, _)| f).collect();
    assert_eq!(failed, ["other.v"]);
    assert!(db.embedder_in("notes", "emb").is_some());

    // A table dropped and created again with another embedding gets the
    // model of its new schema.
    written(&db, "DROP TABLE notes");
    assert!(db.embedder_in("notes", "emb").is_none());
    written(&db, "CREATE TABLE notes (id KEYWORD PRIMARY KEY, body TEXT, emb VECTOR OPTIONS(embed = 'body', model = 'hash:8'))");
    assert_eq!(db.embedder_in("notes", "emb").map(|e| e.dim()), Some(8));
    written(&db, "INSERT INTO notes (id, body) VALUES ('m', 'short')");
    assert_eq!(rows(&db, "SELECT emb FROM notes"), [[Value::Vector(HashEmbedder { dim: 8 }.embed(&["short"]).unwrap().remove(0).into())]]);
}
